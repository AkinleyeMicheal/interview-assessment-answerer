"""Audio capture for interview mode.

SystemAudioCapture — WASAPI loopback (hears the interviewer from the
browser tab).  MicCapture — desktop microphone (mock interview answers).

Both convert to 16 kHz mono int16 PCM, the format ElevenLabs Scribe
Realtime expects.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger("quizbox.audio")

TARGET_RATE = 16000
# Mean |sample| (int16) above this counts as audible — silence/noise floor
# sits well under 100, speech bursts run in the hundreds-to-thousands.
LOUD_RMS = 250


def resample_to_16k_mono(pcm_bytes: bytes, source_rate: int, source_channels: int) -> bytes:
    """Convert interleaved int16 PCM to 16 kHz mono int16 PCM."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if source_channels > 1:
        samples = samples.reshape(-1, source_channels).mean(axis=1)
    if source_rate != TARGET_RATE:
        target_len = int(len(samples) * TARGET_RATE / source_rate)
        if target_len > 0:
            x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
            samples = np.interp(x_new, x_old, samples.astype(np.float64))
        else:
            samples = np.array([], dtype=np.float64)
    return samples.astype(np.int16).tobytes()


class SystemAudioCapture:
    """Captures what the PC is playing via WASAPI loopback."""

    def __init__(self):
        self._running = False
        self._thread: threading.Thread | None = None
        self._callback = None
        self._pa = None
        self._stream = None
        self._rate = 48000
        self._channels = 2
        # Capture-health counters (read by the interview session's health tick)
        self._frames = 0
        self._last_frame_at = 0.0
        self._last_loud_at = 0.0

    @staticmethod
    def find_loopback_device() -> dict | None:
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        try:
            try:
                device = pa.get_default_wasapi_loopback()
                if device:
                    return {
                        "index": device["index"],
                        "name": device["name"],
                        "rate": int(device["defaultSampleRate"]),
                        "channels": int(device["maxInputChannels"]),
                    }
            except Exception as exc:
                log.debug("Default loopback lookup failed: %s", exc)
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("isLoopbackDevice"):
                    return {
                        "index": i,
                        "name": info["name"],
                        "rate": int(info["defaultSampleRate"]),
                        "channels": int(info["maxInputChannels"]),
                    }
        except Exception as exc:
            log.error("Loopback device lookup failed: %s", exc)
        finally:
            pa.terminate()
        return None

    def start(self, callback) -> dict:
        """Start capturing.  ``callback(bytes)`` receives 16 kHz mono PCM."""
        if self._running:
            return {"ok": True, "message": "Already running"}

        import pyaudiowpatch as pyaudio

        device = self.find_loopback_device()
        if device is None:
            return {"ok": False, "error": "No WASAPI loopback device found"}

        self._callback = callback
        self._rate = device["rate"]
        self._channels = device["channels"]
        self._running = True
        self._frames = 0
        self._last_frame_at = 0.0
        self._last_loud_at = 0.0

        def _run():
            try:
                pa = pyaudio.PyAudio()
                self._pa = pa

                def audio_callback(in_data, frame_count, time_info, status):
                    if not self._running:
                        return (None, pyaudio.paComplete)
                    try:
                        if self._callback and in_data:
                            pcm = resample_to_16k_mono(in_data, self._rate, self._channels)
                            if pcm:
                                now = time.monotonic()
                                self._frames += 1
                                self._last_frame_at = now
                                samples = np.frombuffer(pcm, dtype=np.int16)
                                if (samples.size and
                                        float(np.abs(samples.astype(np.float32)).mean()) > LOUD_RMS):
                                    self._last_loud_at = now
                                self._callback(pcm)
                    except Exception as exc:
                        log.error("Audio callback error: %s", exc)
                    return (None, pyaudio.paContinue)

                self._stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=self._channels,
                    rate=self._rate,
                    input=True,
                    input_device_index=device["index"],
                    frames_per_buffer=4800,
                    stream_callback=audio_callback,
                )
                self._stream.start_stream()
                log.info(
                    "Loopback capture started: %s (%d Hz, %d ch)",
                    device["name"], self._rate, self._channels,
                )
                while self._running:
                    if not self._stream.is_active():
                        log.error("Loopback stream became inactive")
                        break
                    threading.Event().wait(0.5)
            except Exception as exc:
                log.error("Loopback capture failed: %s", exc)
                self._running = False

        self._thread = threading.Thread(target=_run, daemon=True, name="loopback-capture")
        self._thread.start()
        return {"ok": True, "device": device["name"], "rate": self._rate}

    def stop(self):
        self._running = False
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        try:
            if self._pa:
                self._pa.terminate()
        except Exception:
            pass
        self._stream = None
        self._pa = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        log.info("Loopback capture stopped")

    def is_running(self) -> bool:
        return self._running

    def activity(self) -> dict:
        """Capture-health snapshot for the live-transcript indicator.

        ``frames`` flowing means the loopback device is alive; ``heard`` means
        audible content (speech/music) was captured at least once.
        """
        now = time.monotonic()
        return {
            "running": self._running,
            "frames": self._frames,
            "heard": self._last_loud_at > 0.0,
            "last_frame_age": (now - self._last_frame_at) if self._last_frame_at else None,
            "last_loud_age": (now - self._last_loud_at) if self._last_loud_at else None,
        }


class MicCapture:
    """Captures the desktop microphone (mock interview)."""

    def __init__(self):
        self._running = False
        self._thread: threading.Thread | None = None
        self._callback = None
        self._pa = None
        self._stream = None
        self._rate = 16000
        self._channels = 1

    @staticmethod
    def find_default_mic() -> dict | None:
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        try:
            device = pa.get_default_input_device_info()
            return {
                "index": device["index"],
                "name": device["name"],
                "rate": int(device["defaultSampleRate"]),
                "channels": min(2, int(device["maxInputChannels"])),
            }
        except Exception as exc:
            log.error("Default mic lookup failed: %s", exc)
        finally:
            pa.terminate()
        return None

    def start(self, callback) -> dict:
        if self._running:
            return {"ok": True, "message": "Already running"}

        import pyaudiowpatch as pyaudio

        device = self.find_default_mic()
        if device is None:
            return {"ok": False, "error": "No microphone found"}

        self._callback = callback
        self._rate = device["rate"]
        self._channels = device["channels"]
        self._running = True

        def _run():
            try:
                pa = pyaudio.PyAudio()
                self._pa = pa

                def audio_callback(in_data, frame_count, time_info, status):
                    if not self._running:
                        return (None, pyaudio.paComplete)
                    try:
                        if self._callback and in_data:
                            pcm = resample_to_16k_mono(in_data, self._rate, self._channels)
                            if pcm:
                                self._callback(pcm)
                    except Exception as exc:
                        log.error("Mic callback error: %s", exc)
                    return (None, pyaudio.paContinue)

                self._stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=self._channels,
                    rate=self._rate,
                    input=True,
                    input_device_index=device["index"],
                    frames_per_buffer=1600,
                    stream_callback=audio_callback,
                )
                self._stream.start_stream()
                log.info("Mic capture started: %s", device["name"])
                while self._running and self._stream and self._stream.is_active():
                    threading.Event().wait(0.5)
            except Exception as exc:
                log.error("Mic capture failed: %s", exc)
                self._running = False

        self._thread = threading.Thread(target=_run, daemon=True, name="mic-capture")
        self._thread.start()
        return {"ok": True, "device": device["name"]}

    def stop(self):
        self._running = False
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        try:
            if self._pa:
                self._pa.terminate()
        except Exception:
            pass
        self._stream = None
        self._pa = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        log.info("Mic capture stopped")

    def is_running(self) -> bool:
        return self._running

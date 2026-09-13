"""Low-latency screen capture primitives.

The stream is intentionally a *latest frame* stream. It never keeps a queue of
captured or JPEG-encoded frames: when a client is slow, the next frame it sees
is the newest one available instead of an increasingly stale one.
"""

from __future__ import annotations

import threading
import time


class SharedFrame:
    """Thread-safe single-slot buffer for the most recently captured frame."""

    def __init__(self):
        self._frame = None
        self._sequence = 0
        self._captured_at = 0.0
        self._lock = threading.Lock()

    def update(self, frame):
        with self._lock:
            self._frame = frame
            self._sequence += 1
            self._captured_at = time.time()
            return self._sequence, self._captured_at

    def get(self):
        """Return a safe copy plus its sequence number and wall-clock timestamp."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy(), self._sequence, self._captured_at

    def clear(self):
        with self._lock:
            self._frame = None
            self._captured_at = 0.0


class ScreenCapture:
    """Capture and JPEG-encode a selected screen region with bounded latency."""

    # A 20 FPS capture / 15 FPS delivery rate keeps interaction responsive while
    # avoiding CPU, Wi-Fi, and browser decode saturation from uncapped MJPEG.
    capture_fps = 20
    stream_fps = 15
    jpeg_quality = 50

    def __init__(self):
        self._frame = SharedFrame()
        self._jpeg = None
        self._jpeg_sequence = 0
        self._jpeg_captured_at = 0.0
        self._jpeg_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._camera = None
        self._region = None  # (left, top, right, bottom)

    def set_region(self, left, top, right, bottom):
        self._region = (left, top, right, bottom)

    def get_region(self):
        return self._region

    def is_configured(self):
        return self._region is not None

    def is_running(self):
        return self._running

    def start(self):
        if self._running:
            return
        if self._region is None:
            raise ValueError("No region selected. Call set_region() first.")

        import dxcam

        self._frame.clear()
        with self._jpeg_lock:
            self._jpeg = None
            self._jpeg_sequence = 0
            self._jpeg_captured_at = 0.0
        self._camera = dxcam.create(output_color="BGR")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        # `grab` is synchronous, so there is no hidden producer queue here.
        # If encoding takes too long, the next iteration grabs a new frame rather
        # than attempting to catch up by preserving old ones.
        import cv2

        interval = 1 / self.capture_fps
        while self._running:
            started_at = time.monotonic()
            try:
                frame = self._camera.grab(region=self._region)
                if frame is not None:
                    sequence, captured_at = self._frame.update(frame)
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    if ok:
                        # A single JPEG is shared by every stream client. Per-client
                        # encoding competes for CPU and makes clients fall behind.
                        with self._jpeg_lock:
                            self._jpeg = encoded.tobytes()
                            self._jpeg_sequence = sequence
                            self._jpeg_captured_at = captured_at
            except Exception:
                # A transient desktop-capture failure should not terminate the feed.
                pass

            remaining = interval - (time.monotonic() - started_at)
            if remaining > 0:
                time.sleep(remaining)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._camera:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None

    def get_frame(self):
        snapshot = self._frame.get()
        return snapshot[0] if snapshot else None

    def get_latest_jpeg(self):
        """Return `(jpeg, sequence, captured_at)` for the newest encoded frame."""
        with self._jpeg_lock:
            if self._jpeg is None:
                return None
            return self._jpeg, self._jpeg_sequence, self._jpeg_captured_at

    def get_frame_jpeg(self, quality=70):
        """Compatibility helper; prefer :meth:`get_latest_jpeg` for streaming."""
        snapshot = self.get_latest_jpeg()
        if snapshot is not None and quality == self.jpeg_quality:
            return snapshot[0]

        # A non-default quality is only used for a one-off image, never the feed.
        frame_snapshot = self._frame.get()
        if frame_snapshot is None:
            return None
        import cv2

        ok, jpeg = cv2.imencode(
            ".jpg", frame_snapshot[0], [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        return jpeg.tobytes() if ok else None

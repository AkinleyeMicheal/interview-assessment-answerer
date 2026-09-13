"""End-to-end checks for the low-latency MJPEG endpoint.

These tests use a real Uvicorn HTTP listener instead of an ASGI test transport,
so multipart framing, response headers, pacing, and JPEG bytes are all covered.
"""

import http.client
import socket
import threading
import time
import unittest

import cv2
import numpy as np
import uvicorn

import server as quizbox_server
from screen_capture import ScreenCapture


PORT = 8765


def publish_frame(value: int):
    """Put a distinguishable JPEG into the application's latest-frame slot."""
    frame = np.full((48, 64, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    sequence, captured_at = quizbox_server.screen._frame.update(frame)
    with quizbox_server.screen._jpeg_lock:
        quizbox_server.screen._jpeg = encoded.tobytes()
        quizbox_server.screen._jpeg_sequence = sequence
        quizbox_server.screen._jpeg_captured_at = captured_at
    return sequence


def read_part(response):
    line = response.readline().strip()
    while not line:
        line = response.readline().strip()
    if line != b"--frame":
        raise AssertionError(f"unexpected MIME boundary: {line!r}")

    headers = {}
    while True:
        line = response.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        key, value = line.decode("ascii").split(":", 1)
        headers[key.lower()] = value.strip()
    body = response.read(int(headers["content-length"]))
    return headers, body


class LiveStreamEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.uvicorn_server = uvicorn.Server(
            uvicorn.Config(quizbox_server.app, host="127.0.0.1", port=PORT, log_level="warning")
        )
        cls.thread = threading.Thread(target=cls.uvicorn_server.run, daemon=True)
        cls.thread.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", PORT), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("Uvicorn did not start")

    @classmethod
    def tearDownClass(cls):
        cls.uvicorn_server.should_exit = True
        cls.thread.join(timeout=5)

    def test_stream_drops_superseded_frames_and_returns_valid_jpeg(self):
        first_sequence = publish_frame(10)
        connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3)
        self.addCleanup(connection.close)
        connection.request("GET", "/stream")
        response = connection.getresponse()

        self.assertEqual(response.status, 200)
        self.assertIn("multipart/x-mixed-replace", response.getheader("Content-Type"))
        self.assertEqual(response.getheader("Cache-Control"), "no-store, no-cache, must-revalidate, max-age=0")
        self.assertEqual(response.getheader("X-Accel-Buffering"), "no")

        first_headers, first_jpeg = read_part(response)
        self.assertEqual(int(first_headers["x-frame-sequence"]), first_sequence)
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(first_jpeg, np.uint8), cv2.IMREAD_COLOR))

        # Frame 2 is deliberately superseded before the next 15 FPS delivery
        # slot. A queue-based stream would show frame 2; ours must show frame 3.
        publish_frame(80)
        newest_sequence = publish_frame(180)
        started_at = time.monotonic()
        second_headers, second_jpeg = read_part(response)
        elapsed = time.monotonic() - started_at

        self.assertEqual(int(second_headers["x-frame-sequence"]), newest_sequence)
        self.assertGreaterEqual(elapsed, 0.04)
        self.assertLess(elapsed, 0.4)
        decoded = cv2.imdecode(np.frombuffer(second_jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertGreater(float(decoded.mean()), 150)

    def test_capture_worker_encodes_only_the_latest_frame(self):
        class FastFakeCamera:
            def __init__(self):
                self.value = 0

            def grab(self, region):
                self.value += 1
                return np.full((32, 32, 3), self.value % 255, dtype=np.uint8)

            def stop(self):
                pass

        capture = ScreenCapture()
        capture.set_region(0, 0, 32, 32)
        capture._camera = FastFakeCamera()
        capture._running = True
        capture._thread = threading.Thread(target=capture._capture_loop, daemon=True)
        capture._thread.start()
        time.sleep(0.18)
        capture.stop()

        jpeg, sequence, captured_at = capture.get_latest_jpeg()
        self.assertGreaterEqual(sequence, 2)
        self.assertLessEqual(sequence, 5)  # capped at capture_fps, never spins freely
        self.assertLess(time.time() - captured_at, 1)
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))

    def test_frame_pull_endpoint_returns_the_newest_jpeg(self):
        publish_frame(30)
        newest_sequence = publish_frame(220)
        connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3)
        self.addCleanup(connection.close)
        connection.request("GET", "/api/screen/frame?nonce=test")
        response = connection.getresponse()
        jpeg = response.read()

        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "image/jpeg")
        self.assertEqual(int(response.getheader("X-Frame-Sequence")), newest_sequence)
        self.assertEqual(response.getheader("Cache-Control"), "no-store, no-cache, must-revalidate, max-age=0")
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertGreater(float(decoded.mean()), 190)


if __name__ == "__main__":
    unittest.main()

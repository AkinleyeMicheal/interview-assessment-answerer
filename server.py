import base64
import logging
import time
import traceback
import threading
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quizbox")

from config import (
    get_settings,
    update_settings,
    rotate_gemini_key,
    get_gemini_key_status,
)
from gemini_client import (
    analyze_image,
    _reinit_client,
    check_groq_status,
    get_provider_status,
)
from screen_capture import ScreenCapture

app = FastAPI(title="Quizbox AI")

screen = ScreenCapture()


class CaptureRequest(BaseModel):
    image: str
    mode: str | None = None
    groq_strategy: str | None = None


class SettingsRequest(BaseModel):
    GEMINI_API_KEY: str | None = None
    GEMINI_API_KEY_2: str | None = None
    GEMINI_API_KEY_3: str | None = None
    GEMINI_API_KEY_4: str | None = None
    GEMINI_API_KEY_5: str | None = None
    MODEL_NAME: str | None = None
    GROQ_API_KEY: str | None = None
    OCR_MODE: str | None = None
    SEARXNG_URL: str | None = None
    SCREEN_REGION_LEFT: int | None = None
    SCREEN_REGION_TOP: int | None = None
    SCREEN_REGION_RIGHT: int | None = None
    SCREEN_REGION_BOTTOM: int | None = None


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/accept")
async def accept_image(req: CaptureRequest):
    settings = get_settings()
    mode = req.mode or settings["OCR_MODE"]
    if mode not in {"groq_vision", "gemini_vision"}:
        raise HTTPException(status_code=400, detail="Unsupported analysis mode")
    log.info(f"Capture request | mode={mode} groq_strategy={req.groq_strategy or 'quick'}")

    try:
        image_bytes = base64.b64decode(req.image)
    except Exception as e:
        log.error(f"Invalid image data: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    t_start = time.time()

    try:
        result = analyze_image(image_bytes, mode, groq_strategy=req.groq_strategy or "quick")
    except ValueError as e:
        log.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        log.error(f"ConnectionError: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        error_str = str(e)
        log.error(f"RuntimeError: {error_str[:200]}")
        raise HTTPException(status_code=503, detail=error_str)
    except Exception as e:
        error_str = str(e)
        log.error(f"Exception: {error_str[:200]}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error: {error_str}")

    elapsed = time.time() - t_start
    result["mode_used"] = mode
    result["response_time"] = f"{elapsed:.1f}s"

    detected = result.get("detected", False)
    provider = result.get("provider", "?")
    log.info(f"Result: detected={detected} provider={provider} time={elapsed:.1f}s")

    return JSONResponse(content=result)


@app.get("/api/settings")
async def get_settings_route():
    return get_settings()


@app.post("/api/settings")
async def update_settings_route(req: SettingsRequest):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    gemini_keys_changed = any(k in updates for k in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"))
    if gemini_keys_changed:
        _reinit_client()
    return update_settings(updates)


@app.get("/api/providers")
async def providers_status():
    providers = get_provider_status()
    groq = check_groq_status()
    providers["groq_vision"]["groq_status"] = groq
    return providers


@app.post("/api/gemini-key/rotate")
async def rotate_gemini_key_route():
    result = rotate_gemini_key()
    if result["rotated"]:
        _reinit_client()
    return result


@app.get("/api/gemini-key/status")
async def gemini_key_status_route():
    return get_gemini_key_status()


@app.get("/stream")
async def video_stream():
    import asyncio

    async def generate():
        # A stream client must only receive the newest available JPEG.  Waiting
        # for the next delivery slot *before* reading the buffer means a slow
        # network/decoder cannot make this generator replay old frames.
        last_sequence = -1
        next_emit_at = time.monotonic()
        interval = 1 / screen.stream_fps
        while True:
            delay = next_emit_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
                continue

            snapshot = screen.get_latest_jpeg()
            if snapshot is None:
                await asyncio.sleep(0.01)
                continue

            jpeg_bytes, sequence, captured_at = snapshot
            if sequence == last_sequence:
                await asyncio.sleep(0.005)
                continue

            last_sequence = sequence
            next_emit_at = time.monotonic() + interval
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg_bytes)}\r\n".encode()
                + f"X-Frame-Sequence: {sequence}\r\n".encode()
                + f"X-Frame-Captured-At: {captured_at:.6f}\r\n\r\n".encode()
                + jpeg_bytes
                + b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            # Do not let a browser, proxy, or reverse proxy cache/buffer the
            # multipart response. These headers are harmless for direct LAN use.
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/screen/frame")
async def screen_frame():
    """Return one current JPEG for the phone's backpressure-aware frame pull."""
    snapshot = screen.get_latest_jpeg()
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Screen capture has not produced a frame yet")

    jpeg_bytes, sequence, captured_at = snapshot
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Frame-Sequence": str(sequence),
            "X-Frame-Captured-At": f"{captured_at:.6f}",
        },
    )


@app.get("/api/screen/status")
async def screen_status():
    region = screen.get_region()
    latest = screen.get_latest_jpeg()
    return {
        "configured": screen.is_configured(),
        "running": screen.is_running(),
        "stream_fps": screen.stream_fps,
        "frame_age_ms": round((time.time() - latest[2]) * 1000) if latest else None,
        "region": {
            "left": region[0],
            "top": region[1],
            "right": region[2],
            "bottom": region[3],
        } if region else None,
    }


@app.post("/api/screen/select")
async def select_screen_region():
    import asyncio

    def do_select():
        import sys
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QRect, Qt, Signal
        from PySide6.QtGui import QColor, QPainter, QPen, QFont
        from PySide6.QtWidgets import QWidget

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        result_box = [None]
        done = threading.Event()

        class Overlay(QWidget):
            selected = Signal(QRect)
            cancelled = Signal()

            def __init__(self, geo):
                super().__init__()
                self._start = None
                self._current = None
                self.setWindowFlags(
                    Qt.WindowType.FramelessWindowHint
                    | Qt.WindowType.WindowStaysOnTopHint
                    | Qt.WindowType.Tool
                )
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
                self.setMouseTracking(True)
                self.setCursor(Qt.CursorShape.CrossCursor)
                self.setGeometry(geo)

            def showEvent(self, e):
                super().showEvent(e)
                self.grabMouse()
                self.grabKeyboard()

            def mousePressEvent(self, e):
                if e.button() == Qt.MouseButton.LeftButton:
                    self._start = e.globalPosition().toPoint()
                    self._current = self._start
                    self.update()

            def mouseMoveEvent(self, e):
                if self._start:
                    self._current = e.globalPosition().toPoint()
                    self.update()

            def mouseReleaseEvent(self, e):
                if e.button() == Qt.MouseButton.LeftButton and self._start:
                    self._current = e.globalPosition().toPoint()
                    rect = QRect(self._start, self._current).normalized()
                    self.releaseMouse()
                    self.releaseKeyboard()
                    self.close()
                    if rect.width() >= 10 and rect.height() >= 10:
                        result_box[0] = (rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())
                    done.set()

            def keyPressEvent(self, e):
                if e.key() == Qt.Key.Key_Escape:
                    self.releaseMouse()
                    self.releaseKeyboard()
                    self.close()
                    done.set()

            def paintEvent(self, _e):
                p = QPainter(self)
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.fillRect(self.rect(), QColor(0, 0, 0, 100))
                if self._start and self._current:
                    sel = QRect(self._start, self._current).normalized()
                    local = sel.translated(-self.geometry().x(), -self.geometry().y())
                    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
                    p.fillRect(local, Qt.GlobalColor.transparent)
                    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                    p.setPen(QPen(QColor(74, 222, 128), 2))
                    p.drawRect(local.adjusted(0, 0, -1, -1))
                    p.setPen(QColor(74, 222, 128))
                    p.setFont(QFont("Consolas", 11))
                    p.drawText(local.x() + 6, local.y() - 6, f"{sel.width()} x {sel.height()}")
                else:
                    p.setPen(QColor(255, 255, 255, 180))
                    p.setFont(QFont("Segoe UI", 14))
                    p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Drag to select a screen region\nPress ESC to cancel")
                p.end()

        screen_geo = app.primaryScreen().geometry()
        overlay = Overlay(screen_geo)
        overlay.show()

        while not done.is_set():
            app.processEvents()
            done.wait(timeout=0.01)

        return result_box[0]

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, do_select)

    if result is None:
        return {"selected": False, "message": "Selection cancelled"}

    left, top, right, bottom = result
    update_settings({
        "SCREEN_REGION_LEFT": left,
        "SCREEN_REGION_TOP": top,
        "SCREEN_REGION_RIGHT": right,
        "SCREEN_REGION_BOTTOM": bottom,
    })

    screen.set_region(left, top, right, bottom)

    if not screen.is_running():
        screen.start()

    return {
        "selected": True,
        "region": {"left": left, "top": top, "right": right, "bottom": bottom},
        "width": right - left,
        "height": bottom - top,
    }


@app.post("/api/screen/start")
async def start_screen_capture():
    settings = get_settings()
    left = settings.get("SCREEN_REGION_LEFT", 0)
    top = settings.get("SCREEN_REGION_TOP", 0)
    right = settings.get("SCREEN_REGION_RIGHT", 0)
    bottom = settings.get("SCREEN_REGION_BOTTOM", 0)

    if right <= left or bottom <= top:
        raise HTTPException(status_code=400, detail="No valid screen region configured. Use /api/screen/select first.")

    screen.set_region(left, top, right, bottom)

    if screen.is_running():
        return {"started": True, "message": "Already running"}

    screen.start()
    return {"started": True, "message": "Screen capture started"}


@app.post("/api/screen/stop")
async def stop_screen_capture():
    screen.stop()
    return {"stopped": True}


app.mount("/static", StaticFiles(directory="static"), name="static")


def _get_lan_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    import os
    cert_path = os.path.join(os.path.dirname(__file__), "cert.pem")
    key_path = os.path.join(os.path.dirname(__file__), "key.pem")

    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        print("No SSL certs found. Generating...")
        from gen_cert import generate_self_signed_cert
        generate_self_signed_cert(os.path.dirname(__file__))

    settings = get_settings()
    left = settings.get("SCREEN_REGION_LEFT", 0)
    top = settings.get("SCREEN_REGION_TOP", 0)
    right = settings.get("SCREEN_REGION_RIGHT", 0)
    bottom = settings.get("SCREEN_REGION_BOTTOM", 0)
    if right > left and bottom > top:
        screen.set_region(left, top, right, bottom)
        try:
            screen.start()
            print(f"  Screen capture active: {right-left}x{bottom-top} at ({left},{top})")
        except Exception as e:
            print(f"  Screen capture not available: {e}")

    ip = _get_lan_ip()
    print(f"\n  Quizbox AI running at:")
    print(f"  https://{ip}:8000")
    print(f"  Scan the QR code in the app header to open on your phone.\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile=cert_path, ssl_keyfile=key_path)

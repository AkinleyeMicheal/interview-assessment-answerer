import base64
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from config import (
    get_settings,
    update_settings,
    rotate_gemini_key,
    get_gemini_key_status,
)
from gemini_client import (
    analyze_image,
    _reinit_client,
    check_ollama_status,
    check_openrouter_status,
    get_provider_status,
)

app = FastAPI(title="Quizbox AI")


class CaptureRequest(BaseModel):
    image: str


class SettingsRequest(BaseModel):
    GEMINI_API_KEY: str | None = None
    GEMINI_API_KEY_2: str | None = None
    GEMINI_API_KEY_3: str | None = None
    CAMERA_URL: str | None = None
    OCR_MODE: str | None = None
    MODEL_NAME: str | None = None
    OLLAMA_MODEL: str | None = None
    OLLAMA_URL: str | None = None
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_MODEL_1: str | None = None
    OPENROUTER_MODEL_2: str | None = None


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/accept")
async def accept_image(req: CaptureRequest):
    settings = get_settings()
    mode = settings["OCR_MODE"]

    if mode == "gemini_vision" and not settings.get("GEMINI_API_KEY") and not settings.get("GEMINI_API_KEY_2") and not settings.get("GEMINI_API_KEY_3"):
        raise HTTPException(status_code=400, detail="No Gemini API keys configured")

    if mode in ("openrouter", "tesseract_openrouter") and not settings.get("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=400, detail="OPENROUTER_API_KEY not configured")

    if mode == "auto" and not settings.get("OPENROUTER_API_KEY") and not settings.get("GEMINI_API_KEY") and not settings.get("GEMINI_API_KEY_2") and not settings.get("GEMINI_API_KEY_3"):
        raise HTTPException(status_code=400, detail="No API keys configured. Set at least one provider.")

    if mode == "antigravity" and not settings.get("GEMINI_API_KEY") and not settings.get("GEMINI_API_KEY_2") and not settings.get("GEMINI_API_KEY_3"):
        raise HTTPException(status_code=400, detail="No Gemini API keys configured for Antigravity")

    try:
        image_bytes = base64.b64decode(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        result = analyze_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        error_str = str(e)
        if "All providers failed" in error_str:
            raise HTTPException(status_code=429, detail="All providers rate-limited. Wait 60 seconds or switch mode.")
        raise HTTPException(status_code=503, detail=error_str)
    except Exception as e:
        error_str = str(e)
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            raise HTTPException(status_code=429, detail="Rate limited. Try again in a moment.")
        raise HTTPException(status_code=500, detail=f"Error: {error_str}")

    result["mode_used"] = mode
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
    ollama = check_ollama_status()
    openrouter = check_openrouter_status()
    providers["ollama"]["connected"] = ollama["connected"]
    providers["ollama"]["models"] = ollama["models"]
    providers["openrouter_1"]["api_status"] = openrouter
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


@app.get("/api/camera-status")
async def camera_status():
    import asyncio
    settings = get_settings()
    url = settings["CAMERA_URL"]

    def check():
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
        ret, _ = cap.read()
        cap.release()
        return ret

    try:
        connected = await asyncio.to_thread(check)
    except Exception:
        connected = False

    return {"connected": connected, "url": url}


@app.get("/api/ollama-status")
async def ollama_status():
    return check_ollama_status()


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

    ip = _get_lan_ip()
    print(f"\n  Quizbox AI running at:")
    print(f"  https://{ip}:8000")
    print(f"  Scan the QR code in the app header to open on your phone.\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile=cert_path, ssl_keyfile=key_path)

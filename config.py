import os
from dotenv import load_dotenv

load_dotenv(override=True)

_settings = {
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
    "GEMINI_API_KEY_2": os.getenv("GEMINI_API_KEY_2", ""),
    "GEMINI_API_KEY_3": os.getenv("GEMINI_API_KEY_3", ""),
    "CAMERA_URL": os.getenv("CAMERA_URL", "http://10.53.59.86:8080/video"),
    "OCR_MODE": os.getenv("OCR_MODE", "openrouter"),
    "MODEL_NAME": os.getenv("MODEL_NAME", "gemini-2.5-flash"),
    "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "granite3.2-vision:2b"),
    "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://localhost:11434"),
    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
    "OPENROUTER_MODEL_1": os.getenv("OPENROUTER_MODEL_1", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
    "OPENROUTER_MODEL_2": os.getenv("OPENROUTER_MODEL_2", "openrouter/free"),
}


def get_settings():
    return dict(_settings)


def update_settings(updates: dict):
    allowed_keys = {
        "GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
        "CAMERA_URL", "OCR_MODE", "MODEL_NAME",
        "OLLAMA_MODEL", "OLLAMA_URL",
        "OPENROUTER_API_KEY", "OPENROUTER_MODEL_1", "OPENROUTER_MODEL_2",
    }
    api_key_fields = {"GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "OPENROUTER_API_KEY"}
    for key, value in updates.items():
        if key in allowed_keys and value is not None:
            if key in api_key_fields and value == "":
                continue
            _settings[key] = value
            os.environ[key] = str(value)
    return get_settings()


# --- Gemini API Key Rotation ---

_gemini_keys: list = []
_gemini_key_index: int = 0


def _refresh_gemini_keys():
    global _gemini_keys, _gemini_key_index
    primary = _settings.get("GEMINI_API_KEY", "")
    k2 = _settings.get("GEMINI_API_KEY_2", "")
    k3 = _settings.get("GEMINI_API_KEY_3", "")
    _gemini_keys = [k for k in [primary, k2, k3] if k]
    if _gemini_key_index >= len(_gemini_keys):
        _gemini_key_index = 0


def get_active_gemini_key() -> str:
    _refresh_gemini_keys()
    if not _gemini_keys:
        return ""
    return _gemini_keys[_gemini_key_index]


def get_gemini_key_index() -> int:
    _refresh_gemini_keys()
    return _gemini_key_index


def get_gemini_key_count() -> int:
    _refresh_gemini_keys()
    return len(_gemini_keys)


def rotate_gemini_key() -> dict:
    global _gemini_key_index
    _refresh_gemini_keys()
    if len(_gemini_keys) <= 1:
        return {
            "rotated": False,
            "message": "Only one key configured, nothing to rotate to.",
            "active_index": _gemini_key_index,
            "total_keys": len(_gemini_keys),
            "active_key_preview": _gemini_keys[_gemini_key_index][:12] + "..." if _gemini_keys else "",
        }
    old_index = _gemini_key_index
    _gemini_key_index = (_gemini_key_index + 1) % len(_gemini_keys)
    return {
        "rotated": True,
        "message": f"Switched from key {old_index + 1} to key {_gemini_key_index + 1}",
        "active_index": _gemini_key_index,
        "total_keys": len(_gemini_keys),
        "active_key_preview": _gemini_keys[_gemini_key_index][:12] + "..." if _gemini_keys else "",
    }


def get_gemini_key_status() -> dict:
    _refresh_gemini_keys()
    keys_info = []
    for i, key in enumerate(_gemini_keys):
        keys_info.append({
            "index": i,
            "preview": key[:12] + "..." if len(key) > 12 else key,
            "active": i == _gemini_key_index,
        })
    return {
        "active_index": _gemini_key_index,
        "total_keys": len(_gemini_keys),
        "keys": keys_info,
    }

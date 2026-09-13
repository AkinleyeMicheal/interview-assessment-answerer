import os
from dotenv import load_dotenv

load_dotenv(override=True)

_settings = {
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
    "GEMINI_API_KEY_2": os.getenv("GEMINI_API_KEY_2", ""),
    "GEMINI_API_KEY_3": os.getenv("GEMINI_API_KEY_3", ""),
    "GEMINI_API_KEY_4": os.getenv("GEMINI_API_KEY_4", ""),
    "GEMINI_API_KEY_5": os.getenv("GEMINI_API_KEY_5", ""),
    "MODEL_NAME": os.getenv("MODEL_NAME", "gemini-3.8-flash"),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
    "OCR_MODE": os.getenv("OCR_MODE", "groq_vision"),
    "SCREEN_REGION_LEFT": int(os.getenv("SCREEN_REGION_LEFT", "0")),
    "SCREEN_REGION_TOP": int(os.getenv("SCREEN_REGION_TOP", "0")),
    "SCREEN_REGION_RIGHT": int(os.getenv("SCREEN_REGION_RIGHT", "0")),
    "SCREEN_REGION_BOTTOM": int(os.getenv("SCREEN_REGION_BOTTOM", "0")),
}


def get_settings():
    return dict(_settings)


def update_settings(updates: dict):
    allowed_keys = {
        "GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
        "GEMINI_API_KEY_4", "GEMINI_API_KEY_5",
        "MODEL_NAME", "GROQ_API_KEY", "OCR_MODE", "SEARXNG_URL",
        "SCREEN_REGION_LEFT", "SCREEN_REGION_TOP", "SCREEN_REGION_RIGHT", "SCREEN_REGION_BOTTOM",
    }
    api_key_fields = {"GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GEMINI_API_KEY_5", "GROQ_API_KEY"}
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
    k4 = _settings.get("GEMINI_API_KEY_4", "")
    k5 = _settings.get("GEMINI_API_KEY_5", "")
    _gemini_keys = [k for k in [primary, k2, k3, k4, k5] if k]
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

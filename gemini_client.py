import base64
import re
import time
import json
import urllib.request
import urllib.error
from google import genai
from google.genai import types
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from config import get_settings, get_active_gemini_key, rotate_gemini_key, get_gemini_key_status


def _preprocess_image(image: Image.Image) -> Image.Image:
    img = image.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.point(lambda x: 0 if x < 140 else 255)
    return img


def _extract_text_ocr(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))

    text1 = pytesseract.image_to_string(image, config="--psm 6").strip()

    preprocessed = _preprocess_image(image)
    text2 = pytesseract.image_to_string(preprocessed, config="--psm 6").strip()

    text3 = pytesseract.image_to_string(preprocessed, config="--psm 4").strip()

    best = max([text1, text2, text3], key=lambda t: len(t))
    return best

QUIZ_PROMPT = """You are an expert exam assistant with deep knowledge across all subjects. Your task is to analyze this image and answer the question shown.

RULES:
1. Ignore all browser UI, toolbars, ads, sidebars, logos, and any non-question content.
2. Focus ONLY on the actual question text and answer options.
3. If no question is visible, respond exactly: NO_QUESTION_DETECTED

FOR MCQs (Multiple Choice Questions):
- Carefully read ALL options (A, B, C, D, etc.) before answering.
- Choose the SINGLE most correct answer based on facts, not opinions.
- If unsure between two options, use reasoning and elimination to pick the best one.
- Always provide the letter AND the text of the correct option.

FOR ESSAY/OPEN-ENDED QUESTIONS:
- Provide a comprehensive, well-structured answer.
- Include key points and relevant details.
- Keep the answer concise but complete.

Format your response EXACTLY as:
QUESTION: [full question text exactly as shown in the image]
TYPE: MCQ or ESSAY
ANSWER: [the correct option letter and text for MCQ, or the answer for ESSAY]
EXPLANATION: [clear explanation of why this is correct and why other options are wrong for MCQ]"""

OCR_TEXT_PROMPT_TEMPLATE = """You are an expert exam assistant. Below is text extracted via OCR from a quiz/test screen.

RULES:
1. Determine if the text contains a question. If no question is found, respond exactly: NO_QUESTION_DETECTED
2. If it is an MCQ, carefully read ALL options and choose the SINGLE most correct answer.
3. If it is an essay/open-ended question, provide a comprehensive answer.

Extracted text:
---
{ocr_text}
---

Format your response EXACTLY as:
QUESTION: [full question text]
TYPE: MCQ or ESSAY
ANSWER: [the correct option letter and text for MCQ, or the answer for ESSAY]
EXPLANATION: [clear explanation of why this is correct and why other options are wrong for MCQ]"""

_client = None
_last_request_time = 0
MIN_INTERVAL = 1.0
COOLDOWN_SECONDS = 60

_provider_status = {}


def _get_client():
    global _client
    if _client is None:
        api_key = get_active_gemini_key()
        if not api_key:
            raise ValueError("No Gemini API keys configured. Set GEMINI_API_KEY in .env or settings.")
        _client = genai.Client(api_key=api_key)
    return _client


def _reinit_client():
    """Reinitialize the Gemini client (forces re-creation with current active key)."""
    global _client
    _client = None


def _gemini_request_with_rotation(func, *args, **kwargs):
    """
    Call a Gemini provider function. On rate-limit error, auto-rotate the key
    and retry once with the new key.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_str = str(e)
        if _is_rate_limit_error(error_str):
            rotation = rotate_gemini_key()
            if rotation["rotated"]:
                _reinit_client()
                try:
                    return func(*args, **kwargs)
                except Exception as retry_e:
                    raise RuntimeError(f"Rate-limited on new key after rotation: {str(retry_e)}") from retry_e
        raise


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


def _parse_response(text: str) -> dict:
    result = {"detected": False, "question": "", "type": "", "answer": "", "explanation": ""}

    if "NO_QUESTION_DETECTED" in text:
        return result

    question_match = re.search(r"QUESTION:\s*(.+?)(?=\nTYPE:|$)", text, re.DOTALL)
    type_match = re.search(r"TYPE:\s*(MCQ|ESSAY)", text, re.IGNORECASE)
    answer_match = re.search(r"ANSWER:\s*(.+?)(?=\nEXPLANATION:|$)", text, re.DOTALL)
    explanation_match = re.search(r"EXPLANATION:\s*(.+)", text, re.DOTALL)

    if question_match:
        result["detected"] = True
        result["question"] = question_match.group(1).strip()
    if type_match:
        result["type"] = type_match.group(1).upper()
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    if explanation_match:
        result["explanation"] = explanation_match.group(1).strip()

    return result


def _mark_cooldown(model_id: str):
    _provider_status[model_id] = {
        "available": False,
        "retry_after": time.time() + COOLDOWN_SECONDS,
    }


def _check_cooldown(model_id: str) -> bool:
    status = _provider_status.get(model_id)
    if status is None:
        return True
    if status["available"]:
        return True
    if time.time() >= status["retry_after"]:
        _provider_status[model_id] = {"available": True, "retry_after": 0}
        return True
    return False


def _is_rate_limit_error(error_str: str) -> bool:
    lower = error_str.lower()
    return (
        "RESOURCE_EXHAUSTED" in error_str
        or "429" in error_str
        or "rate" in lower
        or "too many requests" in lower
        or "quota" in lower
        or "provider_unavailable" in lower
        or "resourceexhausted" in lower
        or "request limit reached" in lower
        or "capacity" in lower
    )


def analyze_image(image_bytes: bytes, mode: str = None) -> dict:
    settings = get_settings()
    if mode is None:
        mode = settings["OCR_MODE"]

    if mode == "antigravity":
        return _analyze_with_antigravity(image_bytes)
    elif mode == "gemini_vision":
        return _gemini_request_with_rotation(_analyze_with_gemini_vision, image_bytes)
    elif mode == "openrouter":
        return _analyze_with_openrouter(image_bytes)
    elif mode == "tesseract_openrouter":
        return _analyze_with_tesseract_openrouter(image_bytes)
    elif mode == "auto":
        return _analyze_with_auto_fallback(image_bytes)
    elif mode == "ollama_vision":
        return _analyze_with_ollama_vision(image_bytes)
    elif mode == "ollama_text":
        return _analyze_with_ollama_text(image_bytes)
    elif mode == "tesseract":
        return _gemini_request_with_rotation(_analyze_with_tesseract_gemini, image_bytes)
    return _analyze_with_openrouter(image_bytes)


def _analyze_with_auto_fallback(image_bytes: bytes) -> dict:
    settings = get_settings()
    errors = []
    last_result = None

    # Step 0: Try Antigravity managed agent (most capable)
    if _check_cooldown("gemini") and settings.get("GEMINI_API_KEY"):
        try:
            result = _analyze_with_antigravity(image_bytes)
            if result.get("detected"):
                result["provider"] = "antigravity"
                return result
            last_result = result
            errors.append("Antigravity: no question detected")
        except Exception as e:
            error_str = str(e)
            if _is_rate_limit_error(error_str):
                _mark_cooldown("gemini")
                rotation = rotate_gemini_key()
                if rotation["rotated"]:
                    _reinit_client()
                errors.append("Antigravity: rate-limited")
            else:
                errors.append(f"Antigravity: {error_str[:80]}")

    # Step 1: Try OCR + OpenRouter text (most reliable)
    if _check_cooldown("openrouter_1") and settings.get("OPENROUTER_API_KEY"):
        try:
            result = _analyze_with_tesseract_openrouter(image_bytes)
            if result.get("detected"):
                result["provider"] = f"ocr+openrouter:{settings.get('OPENROUTER_MODEL_1', '')}"
                return result
            last_result = result
            errors.append("OCR+OpenRouter: no question detected")
        except Exception as e:
            error_str = str(e)
            if _is_rate_limit_error(error_str):
                _mark_cooldown("openrouter_1")
                errors.append("OCR+OpenRouter: rate-limited")
            else:
                errors.append(f"OCR+OpenRouter: {error_str[:80]}")

    # Step 2: Try OpenRouter vision directly
    if _check_cooldown("openrouter_1") and settings.get("OPENROUTER_API_KEY"):
        try:
            result = _analyze_with_openrouter(image_bytes, settings["OPENROUTER_MODEL_1"])
            if result.get("detected"):
                result["provider"] = f"openrouter_vision:{settings['OPENROUTER_MODEL_1']}"
                return result
            last_result = result
            errors.append("OpenRouter vision: no question detected")
        except Exception as e:
            error_str = str(e)
            if _is_rate_limit_error(error_str):
                _mark_cooldown("openrouter_1")
                errors.append("OpenRouter vision: rate-limited")
            else:
                errors.append(f"OpenRouter vision: {error_str[:80]}")

    # Step 3: Try OCR + OpenRouter backup model
    if _check_cooldown("openrouter_2") and settings.get("OPENROUTER_API_KEY"):
        try:
            result = _analyze_with_tesseract_openrouter(image_bytes)
            if result.get("detected"):
                result["provider"] = f"ocr+openrouter:{settings.get('OPENROUTER_MODEL_2', '')}"
                return result
            last_result = result
        except Exception as e:
            error_str = str(e)
            if _is_rate_limit_error(error_str):
                _mark_cooldown("openrouter_2")
                errors.append("OCR+OpenRouter backup: rate-limited")
            else:
                errors.append(f"OCR+OpenRouter backup: {error_str[:80]}")

    # Step 4: Try OCR + Gemini
    if _check_cooldown("gemini") and settings.get("GEMINI_API_KEY"):
        try:
            result = _analyze_with_tesseract_gemini(image_bytes)
            if result.get("detected"):
                result["provider"] = "ocr+gemini"
                return result
            last_result = result
        except Exception as e:
            error_str = str(e)
            if _is_rate_limit_error(error_str):
                _mark_cooldown("gemini")
                rotation = rotate_gemini_key()
                if rotation["rotated"]:
                    _reinit_client()
                errors.append("OCR+Gemini: rate-limited")
            else:
                errors.append(f"OCR+Gemini: {error_str[:80]}")

    # Step 5: Try Gemini vision
    if _check_cooldown("gemini") and settings.get("GEMINI_API_KEY"):
        try:
            result = _analyze_with_gemini_vision(image_bytes)
            if result.get("detected"):
                result["provider"] = "gemini_vision"
                return result
            last_result = result
        except Exception as e:
            error_str = str(e)
            if _is_rate_limit_error(error_str):
                _mark_cooldown("gemini")
                rotation = rotate_gemini_key()
                if rotation["rotated"]:
                    _reinit_client()
                errors.append("Gemini vision: rate-limited")
            else:
                errors.append(f"Gemini vision: {error_str[:80]}")

    # Step 6: Try Ollama text (OCR + local model)
    if _check_cooldown("ollama"):
        try:
            result = _analyze_with_ollama_text(image_bytes)
            if result.get("detected"):
                result["provider"] = "ollama_text"
                return result
            last_result = result
        except Exception as e:
            errors.append(f"Ollama text: {str(e)[:80]}")

    # Step 7: Try Ollama vision
    if _check_cooldown("ollama"):
        try:
            result = _analyze_with_ollama_vision(image_bytes)
            if result.get("detected"):
                result["provider"] = "ollama_vision"
                return result
            last_result = result
        except Exception as e:
            errors.append(f"Ollama vision: {str(e)[:80]}")

    if last_result:
        last_result["provider"] = last_result.get("provider", "auto_fallback")
        return last_result

    raise RuntimeError(f"All providers failed: {'; '.join(errors)}")


def _analyze_with_antigravity(image_bytes: bytes) -> dict:
    settings = get_settings()
    api_key = settings.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Required for Antigravity agent.")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "agent": "antigravity-preview-05-2026",
        "input": [
            {
                "type": "image",
                "data": image_b64,
                "mime_type": "image/jpeg",
            },
            {
                "type": "text",
                "text": QUIZ_PROMPT,
            },
        ],
        "environment": "remote",
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            output_text = ""

            # Try different response structures
            output_text = result.get("output_text", "")
            if not output_text:
                output = result.get("output", {})
                if isinstance(output, dict):
                    output_text = output.get("text", "")
            if not output_text:
                steps = result.get("steps", [])
                for step in steps:
                    if isinstance(step, dict):
                        # Try step.text
                        if step.get("type") == "text" and step.get("text"):
                            output_text = step["text"]
                            break
                        # Try step.content[0].text (Antigravity format)
                        content = step.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    output_text = item.get("text", "")
                                    if output_text:
                                        break
                        if output_text:
                            break
            if not output_text:
                output_text = json.dumps(result)

            parsed = _parse_response(output_text)
            parsed["provider"] = "antigravity"
            return parsed
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        if e.code == 429:
            raise RuntimeError(f"Antigravity rate-limited (429): {body}")
        raise RuntimeError(f"Antigravity HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot connect to Antigravity API. Error: {e}")


def _analyze_with_openrouter(image_bytes: bytes, model: str = None) -> dict:
    settings = get_settings()
    api_key = settings.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Configure it in .env or settings.")

    if model is None:
        model = settings.get("OPENROUTER_MODEL_1", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUIZ_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://quizbox-ai.local",
            "X-Title": "Quizbox AI",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

            if "error" in result:
                err_msg = result["error"].get("message", str(result["error"]))
                err_code = result["error"].get("code", 500)
                raise RuntimeError(f"OpenRouter error {err_code}: {err_msg}")

            content = result["choices"][0]["message"]["content"]
            parsed = _parse_response(content)
            parsed["provider"] = f"openrouter:{model}"
            return parsed
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot connect to OpenRouter. Error: {e}")


def _analyze_with_tesseract_openrouter(image_bytes: bytes) -> dict:
    settings = get_settings()
    api_key = settings.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    model = settings.get("OPENROUTER_MODEL_1", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")

    extracted_text = _extract_text_ocr(image_bytes)

    if not extracted_text or len(extracted_text) < 10:
        try:
            parsed = _analyze_with_openrouter(image_bytes)
            parsed["provider"] = f"openrouter_vision:{model}"
            return parsed
        except Exception:
            return {"detected": False, "question": "", "type": "", "answer": "", "explanation": "", "provider": "tesseract+openrouter"}

    prompt = OCR_TEXT_PROMPT_TEMPLATE.format(ocr_text=extracted_text)

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://quizbox-ai.local",
            "X-Title": "Quizbox AI",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

            if "error" in result:
                err_msg = result["error"].get("message", str(result["error"]))
                err_code = result["error"].get("code", 500)
                raise RuntimeError(f"OpenRouter error {err_code}: {err_msg}")

            content = result["choices"][0]["message"]["content"]
            parsed = _parse_response(content)
            parsed["provider"] = f"tesseract+openrouter:{model}"
            return parsed
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot connect to OpenRouter. Error: {e}")


def _analyze_with_gemini_vision(image_bytes: bytes) -> dict:
    _rate_limit()
    client = _get_client()
    settings = get_settings()

    response = client.models.generate_content(
        model=settings["MODEL_NAME"],
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            QUIZ_PROMPT,
        ],
    )

    result = _parse_response(response.text)
    result["provider"] = "gemini"
    return result


def _analyze_with_ollama_vision(image_bytes: bytes) -> dict:
    settings = get_settings()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": settings["OLLAMA_MODEL"],
        "prompt": QUIZ_PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
        },
    }

    url = f"{settings['OLLAMA_URL']}/api/generate"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            parsed = _parse_response(result.get("response", ""))
            parsed["provider"] = "ollama"
            return parsed
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot connect to Ollama at {settings['OLLAMA_URL']}. Is it running? Error: {e}")


def _analyze_with_ollama_text(image_bytes: bytes) -> dict:
    settings = get_settings()

    extracted_text = _extract_text_ocr(image_bytes)

    if not extracted_text or len(extracted_text) < 10:
        try:
            parsed = _analyze_with_ollama_vision(image_bytes)
            parsed["provider"] = "ollama_vision"
            return parsed
        except Exception:
            return {"detected": False, "question": "", "type": "", "answer": "", "explanation": "", "provider": "tesseract+ollama"}

    prompt = OCR_TEXT_PROMPT_TEMPLATE.format(ocr_text=extracted_text)

    payload = {
        "model": settings["OLLAMA_MODEL"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
        },
    }

    url = f"{settings['OLLAMA_URL']}/api/generate"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            parsed = _parse_response(result.get("response", ""))
            parsed["provider"] = "tesseract+ollama"
            return parsed
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot connect to Ollama at {settings['OLLAMA_URL']}. Is it running? Error: {e}")


def _analyze_with_tesseract_gemini(image_bytes: bytes) -> dict:
    _rate_limit()
    client = _get_client()
    settings = get_settings()

    extracted_text = _extract_text_ocr(image_bytes)

    if not extracted_text or len(extracted_text) < 10:
        return {"detected": False, "question": "", "type": "", "answer": "", "explanation": "", "provider": "tesseract+gemini"}

    prompt = OCR_TEXT_PROMPT_TEMPLATE.format(ocr_text=extracted_text)

    response = client.models.generate_content(
        model=settings["MODEL_NAME"],
        contents=[prompt],
    )

    result = _parse_response(response.text)
    result["provider"] = "tesseract+gemini"
    return result


def get_provider_status() -> dict:
    settings = get_settings()
    now = time.time()

    providers = {
        "antigravity": {
            "name": "Antigravity (Google Agent)",
            "model": "antigravity-preview-05-2026",
            "configured": bool(settings.get("GEMINI_API_KEY")),
            "available": _check_cooldown("gemini"),
            "cooldown_remaining": 0,
        },
        "openrouter_1": {
            "name": "OpenRouter (Primary)",
            "model": settings.get("OPENROUTER_MODEL_1", ""),
            "configured": bool(settings.get("OPENROUTER_API_KEY")),
            "available": _check_cooldown("openrouter_1"),
            "cooldown_remaining": 0,
        },
        "openrouter_2": {
            "name": "OpenRouter (Backup)",
            "model": settings.get("OPENROUTER_MODEL_2", ""),
            "configured": bool(settings.get("OPENROUTER_API_KEY")),
            "available": _check_cooldown("openrouter_2"),
            "cooldown_remaining": 0,
        },
        "gemini": {
            "name": "Gemini (Fallback)",
            "model": settings.get("MODEL_NAME", ""),
            "configured": bool(settings.get("GEMINI_API_KEY")),
            "available": _check_cooldown("gemini"),
            "cooldown_remaining": 0,
        },
        "ollama": {
            "name": "Ollama (Local)",
            "model": settings.get("OLLAMA_MODEL", ""),
            "configured": True,
            "available": _check_cooldown("ollama"),
            "cooldown_remaining": 0,
        },
    }

    for key in ["antigravity", "openrouter_1", "openrouter_2", "gemini", "ollama"]:
        status = _provider_status.get(key)
        if status and not status["available"]:
            remaining = max(0, int(status["retry_after"] - now))
            providers[key]["cooldown_remaining"] = remaining
            providers[key]["available"] = remaining == 0

    return providers


def check_ollama_status() -> dict:
    settings = get_settings()
    url = f"{settings['OLLAMA_URL']}/api/tags"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            return {"connected": True, "models": models}
    except Exception:
        return {"connected": False, "models": []}


def check_openrouter_status() -> dict:
    settings = get_settings()
    api_key = settings.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return {"configured": False, "valid": False, "error": "No API key set"}

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            key_data = data.get("data", {})
            return {
                "configured": True,
                "valid": True,
                "label": key_data.get("label", ""),
                "usage": key_data.get("usage", 0),
                "limit_remaining": key_data.get("limit_remaining"),
                "is_free_tier": key_data.get("is_free_tier", True),
            }
    except urllib.error.HTTPError as e:
        return {"configured": True, "valid": False, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"configured": True, "valid": False, "error": str(e)}

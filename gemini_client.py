"""The two, intentionally separate, vision-analysis pipelines for Quizbox."""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request

from google import genai
from google.genai import types

from config import get_active_gemini_key, get_settings, rotate_gemini_key


GROQ_VISION_MODEL = "qwen/qwen3.6-27b"
GROQ_SEARCH_MODEL = "groq/compound-mini"
GROQ_TEXT_FALLBACK_MODEL = "openai/gpt-oss-20b"


GROQ_EXTRACTION_PROMPT = """Read this quiz screenshot and extract the question or instruction along with every visible answer option.

This includes: direct questions, instructions like "pick the odd one out", prompts to write code, fill-in-the-blank, true/false, matching, or any other task. If there is nothing to answer or do, respond exactly with NO_QUESTION_DETECTED.

Do not solve the task yet. Respond exactly in this format:
QUESTION: [the full question, instruction, or task text exactly as written]
TYPE: MCQ or ESSAY
OPTIONS: [for an MCQ, preserve all options including their letters; for anything else write N/A]"""


GROQ_QUICK_PROMPT = """You are an expert exam assistant. Analyze this quiz screenshot and answer the question or complete the instruction shown.

Ignore browser UI, toolbars, ads, logos, timers, and unrelated content. If there is nothing to answer or do, respond exactly with NO_QUESTION_DETECTED.

This includes: direct questions, "pick the odd one out", "write code to...", "what is the output of...", true/false, fill-in-the-blank, or any other task.

For an MCQ, read every option and select the one correct answer.
For any other task (essay, code writing, explanation, odd-one-out, etc.), provide a thorough, complete answer. Do not truncate or summarize — give the full answer as you would in an exam.

Respond exactly in this format:
QUESTION: [the full question, instruction, or task text]
TYPE: MCQ or ESSAY
OPTIONS: [all options with letters, or N/A]
ANSWER: [the correct letter and exact option text for MCQ; for other tasks, a complete well-developed answer]
EXPLANATION: [clear brief justification]"""


GROQ_SEARCH_PROMPT = """You are an exam-answering assistant. Use the enabled web-search tool to verify the answer to the supplied task before answering. Use the search result to resolve factual uncertainty; do not invent sources or change the supplied options.

For an MCQ, return the single correct letter and its exact option text.
For any other task (essay, code writing, explanation, odd-one-out, etc.), provide a thorough, complete answer. Do not truncate or summarize.

Return exactly this format:
QUESTION: [repeat the supplied task]
TYPE: MCQ or ESSAY
OPTIONS: [repeat the supplied options, or N/A]
ANSWER: [the answer as described above]
EXPLANATION: [brief justification grounded in the verified information]

Task:
{question}

Task type: {question_type}

Options:
{options}"""


GEMINI_VISION_PROMPT = """You are an expert exam assistant. Analyze this quiz screenshot and answer the question or complete the instruction shown.

Ignore browser UI, toolbars, ads, logos, timers, and unrelated content. If there is nothing to answer or do, respond exactly with NO_QUESTION_DETECTED.

This includes: direct questions, "pick the odd one out", "write code to...", "what is the output of...", true/false, fill-in-the-blank, or any other task.

For an MCQ, read every option and select the one correct answer.
For any other task (essay, code writing, explanation, odd-one-out, etc.), provide a thorough, complete answer. Do not truncate or summarize — give the full answer as you would in an exam.

Respond exactly in this format:
QUESTION: [the full question, instruction, or task text]
TYPE: MCQ or ESSAY
OPTIONS: [all options with letters, or N/A]
ANSWER: [the correct letter and exact option text for MCQ; for other tasks, a complete well-developed answer]
EXPLANATION: [clear brief justification]"""


def _parse_response(text: str) -> dict:
    """Parse the shared, labelled answer format without losing multi-line options."""
    result = {
        "detected": False,
        "question": "",
        "type": "",
        "options": "",
        "answer": "",
        "explanation": "",
    }
    if not text:
        return result

    # Models occasionally decorate labels with Markdown emphasis.
    text = re.sub(r"\*{1,2}(QUESTION|TYPE|OPTIONS|ANSWER|EXPLANATION)\*{1,2}\s*:", r"\1:", text, flags=re.I)
    if "NO_QUESTION_DETECTED" in text.upper():
        return result

    labels = "QUESTION|TYPE|OPTIONS|ANSWER|EXPLANATION"

    def value(label: str) -> str:
        match = re.search(
            rf"(?:^|\n){label}:\s*(.*?)(?=\n(?:{labels}):|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    result["question"] = value("QUESTION")
    result["options"] = value("OPTIONS")
    result["answer"] = value("ANSWER")
    result["explanation"] = value("EXPLANATION")
    response_type = value("TYPE").upper()
    if response_type in {"MCQ", "ESSAY"}:
        result["type"] = response_type
    elif result["options"] and result["options"].upper() != "N/A":
        result["type"] = "MCQ"
    elif result["question"]:
        result["type"] = "ESSAY"
    result["detected"] = bool(result["question"])
    return result


def _groq_chat(model: str, messages: list[dict], *, max_tokens: int, timeout: int, compound: bool = False) -> tuple[str, dict]:
    """Make one Groq Chat Completions request and preserve tool metadata."""
    api_key = get_settings().get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if compound:
        # Compound Mini is low-latency and permits one tool call. Restricting it
        # to web_search makes this stage independent of any local search service.
        payload["compound_custom"] = {"tools": {"enabled_tools": ["web_search"]}}

    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Quizbox/1.0",
            "Groq-Model-Version": "latest",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Groq request failed (HTTP {error.code}): {detail}") from error
    except urllib.error.URLError as error:
        raise ConnectionError(f"Unable to reach Groq: {error.reason}") from error

    try:
        message = result["choices"][0]["message"]
        return message.get("content", ""), message
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Groq returned an unexpected response") from error


def _call_groq_vision(image_bytes: bytes) -> str:
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    content, _message = _groq_chat(
        GROQ_VISION_MODEL,
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": GROQ_EXTRACTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
        max_tokens=900,
        timeout=30,
    )
    return content


def _call_groq_quick(image_bytes: bytes) -> str:
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    content, _message = _groq_chat(
        GROQ_VISION_MODEL,
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": GROQ_QUICK_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
        max_tokens=2000,
        timeout=30,
    )
    return content


def _sources_from_tools(message: dict) -> list[dict]:
    """Expose Groq Compound search URLs when its tool response provides them."""
    sources = []
    seen = set()
    for tool in message.get("executed_tools") or []:
        output = tool.get("output", "") if isinstance(tool, dict) else ""
        for url in re.findall(r"https?://[^\s\]\)\}\",]+", output):
            url = url.rstrip(".,;")
            if url not in seen:
                seen.add(url)
                sources.append({"title": "Groq web search result", "uri": url})
    return sources


def _analyze_with_groq_quick(image_bytes: bytes) -> dict:
    """One-call Groq Vision: reads the screenshot and answers in a single request."""
    output = _call_groq_quick(image_bytes)
    parsed = _parse_response(output)
    parsed["provider"] = "groq_vision"
    parsed["answer_mode"] = "quick"
    parsed["search_used"] = False
    parsed["sources"] = []
    return parsed


def _analyze_with_groq_verified(image_bytes: bytes) -> dict:
    """Two-stage: extract with Groq Vision, then verify with Groq web search."""
    extracted = _parse_response(_call_groq_vision(image_bytes))
    if not extracted["detected"]:
        extracted["provider"] = "groq_vision"
        extracted["answer_mode"] = "verified"
        extracted["sources"] = []
        extracted["search_used"] = False
        return extracted

    prompt = GROQ_SEARCH_PROMPT.format(
        question=extracted["question"],
        question_type=extracted["type"] or "ESSAY",
        options=extracted["options"] or "N/A",
    )
    search_used = False
    search_error = ""
    fallback_reason = ""
    try:
        answer_text, message = _groq_chat(
            GROQ_SEARCH_MODEL,
            [{"role": "user", "content": prompt}],
            max_tokens=1500,
            timeout=45,
            compound=True,
        )
        search_used = bool(message.get("executed_tools"))
    except (ConnectionError, RuntimeError) as search_failure:
        fallback_prompt = prompt.replace(
            "Use the enabled web-search tool to verify the answer to the supplied task before answering.",
            "Answer from your own knowledge; web verification is currently unavailable.",
        )
        answer_text, message = _groq_chat(
            GROQ_TEXT_FALLBACK_MODEL,
            [{"role": "user", "content": fallback_prompt}],
            max_tokens=1500,
            timeout=30,
        )
        search_used = False
        search_error = str(search_failure)[:200]
        fallback_reason = "Groq search unavailable — answered from model knowledge"
    answered = _parse_response(answer_text)

    answered["detected"] = True
    answered["question"] = extracted["question"]
    answered["type"] = extracted["type"] or answered["type"]
    answered["options"] = extracted["options"]
    if not answered["answer"]:
        raise RuntimeError("Groq search completed but did not return a labelled answer")
    answered["provider"] = "groq_vision"
    answered["answer_mode"] = "verified"
    answered["sources"] = _sources_from_tools(message)
    answered["search_used"] = search_used
    if search_error:
        answered["search_error"] = search_error
    if fallback_reason:
        answered["fallback_reason"] = fallback_reason
    return answered


# --- Gemini Vision ---

_client = None
_last_request_time = 0.0
MIN_INTERVAL = 1.0


def _get_client():
    global _client
    if _client is None:
        api_key = get_active_gemini_key()
        if not api_key:
            raise ValueError("No Gemini API keys configured")
        _client = genai.Client(api_key=api_key)
    return _client


def _reinit_client():
    global _client
    _client = None


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


def _is_rate_limit_error(error_text: str) -> bool:
    lower = error_text.lower()
    return any(marker in lower for marker in ("resource_exhausted", "429", "503", "unavailable", "rate", "quota", "high demand"))


def _gemini_request_with_rotation(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as error:
        if _is_rate_limit_error(str(error)):
            rotation = rotate_gemini_key()
            if rotation["rotated"]:
                _reinit_client()
                return func(*args, **kwargs)
        raise


def _analyze_with_gemini_vision(image_bytes: bytes) -> dict:
    _rate_limit()
    response = _get_client().models.generate_content(
        model=get_settings()["MODEL_NAME"],
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            GEMINI_VISION_PROMPT,
        ],
    )
    result = _parse_response(response.text)
    result["provider"] = "gemini_vision"
    result["sources"] = []
    return result


def analyze_image(image_bytes: bytes, mode: str | None = None, groq_strategy: str = "quick") -> dict:
    mode = mode or get_settings()["OCR_MODE"]
    if mode == "groq_vision":
        if groq_strategy == "verified":
            return _analyze_with_groq_verified(image_bytes)
        return _analyze_with_groq_quick(image_bytes)
    if mode == "gemini_vision":
        return _gemini_request_with_rotation(_analyze_with_gemini_vision, image_bytes)
    raise ValueError(f"Unsupported analysis mode: {mode}")


def check_groq_status() -> dict:
    """Check the configured key without spending tokens on an inference request."""
    api_key = get_settings().get("GROQ_API_KEY", "")
    if not api_key:
        return {"configured": False, "valid": False, "error": "No API key set"}
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "Quizbox/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            model_ids = {item.get("id") for item in json.loads(response.read().decode("utf-8")).get("data", [])}
        return {
            "configured": True,
            "valid": True,
            "vision_available": GROQ_VISION_MODEL in model_ids,
            "search_available": GROQ_SEARCH_MODEL in model_ids,
        }
    except urllib.error.HTTPError as error:
        return {"configured": True, "valid": False, "error": f"HTTP {error.code}"}
    except Exception as error:
        return {"configured": True, "valid": False, "error": str(error)}


def get_provider_status() -> dict:
    settings = get_settings()
    return {
        "groq_vision": {
            "name": "Groq Vision + web verification",
            "model": f"{GROQ_VISION_MODEL} → {GROQ_SEARCH_MODEL}",
            "configured": bool(settings.get("GROQ_API_KEY")),
            "available": True,
            "cooldown_remaining": 0,
        },
        "gemini_vision": {
            "name": "Gemini Vision",
            "model": settings.get("MODEL_NAME", ""),
            "configured": bool(get_active_gemini_key()),
            "available": True,
            "cooldown_remaining": 0,
        },
    }

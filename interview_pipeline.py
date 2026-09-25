"""Interview pipeline: question detection + answer generation (live mode),
mock interview question generation + answer evaluation (mock mode),
screen fusion (coding copilot).

All functions are synchronous and call Groq or Gemini directly — run them
from a worker thread so they never block the WebSocket event loop.

Provider selection: every call accepts an optional ``ProviderRouter``. Context
(history / stashed problem text / transcript) is rendered into the prompt
before the router is consulted, so a manual mid-session switch or an automatic
failover hands the incoming provider the identical payload.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time

from gemini_client import (
    GROQ_CODE_MODEL,
    GROQ_TEXT_FALLBACK_MODEL,
    GROQ_VISION_MODEL,
    _groq_chat,
    get_active_gemini_key,
    get_settings,
)
from provider_router import DEFAULT_ROUTER, ProviderRouter

log = logging.getLogger("quizbox.interview")


def _chat(model: str, messages: list[dict], max_tokens: int, timeout: int = 30) -> str:
    """_groq_chat with retries — Groq occasionally resets connections."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            content, _message = _groq_chat(model, messages, max_tokens=max_tokens, timeout=timeout)
            return content
        except Exception as exc:
            last_error = exc
            log.warning("Groq call failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(0.8 * (attempt + 1))
    raise last_error if last_error else RuntimeError("Groq call failed")


def _gemini_chat(messages: list[dict], max_tokens: int, timeout: int = 30) -> str:
    """Gemini text chat over an OpenAI-style message list.

    The system message becomes Gemini's system instruction; every other
    message is concatenated in order, so the SAME context payload that the
    Groq path received reaches Gemini unchanged (that is the whole point of
    the provider router: context is built before the provider is chosen).
    ``timeout`` is accepted for signature parity; the genai client manages
    its own transport timeouts.
    """
    from google.genai import types as genai_types
    from gemini_client import _gemini_request_with_rotation, _get_client, _rate_limit

    system_parts = [str(m.get("content", "")) for m in messages if m.get("role") == "system"]
    body_parts = [
        str(m.get("content", ""))
        for m in messages
        if m.get("role") != "system" and str(m.get("content", "")).strip()
    ]
    body = "\n\n".join(p for p in body_parts if p)
    config_kwargs: dict = {"max_output_tokens": max_tokens}
    if system_parts:
        config_kwargs["system_instruction"] = "\n\n".join(system_parts)

    _rate_limit()
    response = _gemini_request_with_rotation(
        lambda: _get_client().models.generate_content(
            model=get_settings()["MODEL_NAME"],
            contents=body,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
    )
    return (response.text or "").strip()


def _history_block(history: list[dict] | None, limit: int = 6) -> str:
    """Render recent interview turns so every provider sees identical context."""
    items = list(history or [])[-limit:]
    lines: list[str] = []
    for item in items:
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if question:
            lines.append(f"Interviewer: {question}")
        if answer:
            lines.append(f"Candidate: {answer}")
    if not lines:
        return ""
    return (
        "So far in this interview (stay consistent with everything that "
        "has already been said):\n" + "\n".join(lines)
    )

LIVE_SYSTEM_PROMPT = """You are an expert interview coach sitting beside a candidate during a live job interview. You will be given the latest transcript of what the interviewer just said.

Your job:
1. Decide whether the interviewer's latest statement is a question or a request for information that the candidate should answer.
2. If it is, answer it the way the candidate should say it out loud.

IMPORTANT: interview questions often do NOT end with a question mark and do NOT start with question words. Treat all of these as questions:
- "Tell me about yourself."
- "Walk me through your resume."
- "I'd like to hear about a challenging project you worked on."
- "What sort of experience do you have with Python?" (or "Experience with Python?")
- "So how did you handle that?"
- "Sell me this pen." / "How would you approach...?" / "Give me an example of..."
- Any request to explain, describe, demonstrate, compare, design, calculate, or opine.

These are NOT questions (respond with IS_QUESTION: NO):
- Encouragement or backchannel: "Go on.", "Interesting.", "I see.", "Right.", "Okay."
- Thanks: "Thanks for that.", "Appreciate it."
- Transitions or instructions: "Let's move on to the next topic.", "Let's do a quick coding question." (but "write code to ..." IS a question)
- Small talk with no information request: "How's your day going so far?" IS a question though.

Answer style when it IS a question:
- Full spoken-style response: conversational, natural, as if the candidate is talking right now.
- Use first person, concrete examples, and a clear structure (situation -> action -> result for behavioral questions).
- Never truncate. Never use markdown, bullets, or headings — this is spoken aloud. 3-6 sentences for most questions, longer for technical or essay-style questions.
- For technical/coding questions: give the approach, then the solution, then a brief example.
- If the statement is a request to do something unrelated to answering (e.g. "please maximize this window"), still answer as the candidate acknowledging and doing it briefly.

Two extra judgements:
- INCOMPLETE: YES when the transcript is clearly a fragment of a longer question that got cut off mid-thought — it ends with a trailing conjunction or preposition ("and then...", "what about the..."), stops mid-sentence without any ending, or is obviously only the first half of a request. NO when it reads as a complete statement or question.
- NEEDS_SCREEN: YES when the question is about code, an algorithm, debugging, a specific line of code, a diagram, a screenshot, or anything you would need to SEE to answer it well. NO for behavioral, motivational, or general knowledge questions.

Respond in exactly this format:
IS_QUESTION: YES or NO
INCOMPLETE: YES or NO
NEEDS_SCREEN: YES or NO
QUESTION: <the question restated naturally, or N/A>
ANSWER: <the full spoken answer, or N/A>"""

MOCK_QUESTION_PROMPT = """You are a professional interviewer conducting a mock {role} interview for a candidate with {level} experience.

Generate ONE realistic, specific interview question. Vary the type as the interview progresses:
- Early: icebreaker / background ("Tell me about yourself", "Why this role?")
- Middle: behavioral (STAR-worthy), technical/role-specific, situational
- Later: deeper probes, system design, case-style, or curveballs

Question #{question_number}. Do not repeat questions already asked.
{previous}
Return ONLY the question text, nothing else, no quotes, no numbering."""

MOCK_EVALUATION_PROMPT = """You are a strict but fair interview coach evaluating a candidate's spoken answer.

Interviewer asked: {question}
Candidate answered: {answer}

Evaluate the answer:
- Relevance, structure (STAR for behavioral), specificity with concrete examples, clarity, depth, and conciseness.
- If the answer is empty, too short to evaluate, or not actually addressing the question, score accordingly.

Respond in exactly this format:
SCORE: <integer 0-10>
FEEDBACK: <2-3 sentences, conversational, what worked and what to improve>
IMPROVED_ANSWER: <a model answer the candidate could learn from, spoken style>"""


def _parse_interview_response(text: str) -> dict:
    """Parse IS_QUESTION / INCOMPLETE / NEEDS_SCREEN / QUESTION / ANSWER blocks."""
    result = {"is_question": False, "incomplete": False, "needs_screen": False,
              "question": "", "answer": ""}
    if not text:
        return result

    match = re.search(
        r"IS_QUESTION:\s*(YES|NO)", text, re.IGNORECASE
    )
    if match:
        result["is_question"] = match.group(1).strip().upper() == "YES"

    incomplete_match = re.search(r"(?<!\w)INCOMPLETE:\s*(YES|NO)", text, re.IGNORECASE)
    if incomplete_match:
        result["incomplete"] = incomplete_match.group(1).strip().upper() == "YES"

    needs_screen_match = re.search(r"(?<!\w)NEEDS_SCREEN:\s*(YES|NO)", text, re.IGNORECASE)
    if needs_screen_match:
        result["needs_screen"] = needs_screen_match.group(1).strip().upper() == "YES"

    q_match = re.search(
        r"(?<!IS_)^QUESTION:\s*(.*?)(?=^ANSWER:|\Z)", text, re.IGNORECASE | re.DOTALL | re.MULTILINE
    )
    a_match = re.search(r"(?<!IS_)^ANSWER:\s*(.*)\Z", text, re.IGNORECASE | re.DOTALL | re.MULTILINE)

    if q_match:
        result["question"] = q_match.group(1).strip()
    if a_match:
        result["answer"] = a_match.group(1).strip()

    if result["question"].upper() == "N/A":
        result["question"] = ""
    if result["answer"].upper() == "N/A":
        result["answer"] = ""

    if not match:
        if result["answer"] or result["question"]:
            result["is_question"] = True
        else:
            # No markers at all: model answered directly (format drift).
            # Surface the text rather than dropping it.
            stripped = text.strip()
            if stripped:
                result["is_question"] = True
                result["answer"] = stripped

    return result


def detect_and_answer(
    transcript: str,
    history: list[dict] | None = None,
    router: ProviderRouter | None = None,
) -> dict:
    """Detect whether an interviewer statement is a question and answer it.

    ``history`` (everything said so far this session) is rendered into the
    prompt BEFORE any provider is chosen, so a mid-session switch or an
    automatic failover hands the incoming model the full conversation.

    Returns {"is_question": bool, "incomplete": bool, "needs_screen": bool,
    "question": str, "answer": str, "provider": str}.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return {"is_question": False, "incomplete": False, "needs_screen": False,
                "question": "", "answer": ""}

    messages = [
        {"role": "system", "content": LIVE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Interviewer just said:\n\"{transcript}\""},
    ]
    context = _history_block(history)
    if context:
        messages.insert(1, {"role": "user", "content": context})

    router = router or DEFAULT_ROUTER
    calls = {
        "groq": lambda: _chat(GROQ_TEXT_FALLBACK_MODEL, messages, max_tokens=2000, timeout=30),
        "gemini": lambda: _gemini_chat(messages, max_tokens=2000, timeout=30),
    }
    try:
        content, served = router.run(calls)
    except Exception as exc:
        log.error("detect_and_answer failed: %s", exc)
        return {"is_question": False, "incomplete": False, "needs_screen": False,
                "question": "", "answer": "", "error": str(exc)}

    result = _parse_interview_response(content)
    result["provider"] = served
    log.info(
        "detect_and_answer: provider=%s is_question=%s q=%.60r answer=%.60r",
        served, result["is_question"], result["question"], result["answer"],
    )
    return result


def generate_mock_question(
    role: str = "software engineer",
    level: str = "mid-level",
    question_number: int = 1,
    previous_qa: list[dict] | None = None,
    router: ProviderRouter | None = None,
) -> str:
    """Generate the next mock interview question.

    ``previous_qa`` (every question already asked + answers given) is rendered
    into the prompt before the router picks a provider, so switching models
    mid-session never loses the interview's momentum.
    """
    previous = ""
    if previous_qa:
        recent = previous_qa[-4:]
        lines = ["Already asked in this session:"]
        for item in recent:
            lines.append(f"- Q: {item.get('question', '')}")
            lines.append(f"  A: {item.get('answer', '')[:200]}")
        previous = "\n".join(lines) + "\n"

    messages = [
        {"role": "system", "content": "You generate interview questions. Return only the question."},
        {"role": "user", "content": MOCK_QUESTION_PROMPT.format(
            role=role, level=level, question_number=question_number, previous=previous,
        )},
    ]
    router = router or DEFAULT_ROUTER
    calls = {
        "groq": lambda: _chat(GROQ_TEXT_FALLBACK_MODEL, messages, max_tokens=800, timeout=25),
        "gemini": lambda: _gemini_chat(messages, max_tokens=800, timeout=25),
    }
    try:
        content, _served = router.run(calls)
    except Exception as exc:
        log.error("generate_mock_question failed: %s", exc)
        return "Tell me about yourself and your background."
    question = content.strip().strip('"').strip()
    return question or "Tell me about yourself."


def evaluate_mock_answer(
    question: str,
    answer: str,
    history: list[dict] | None = None,
    router: ProviderRouter | None = None,
) -> dict:
    """Score a mock interview answer. Returns {"score": int, "feedback": str, "improved_answer": str}."""
    answer = (answer or "").strip()
    if not answer:
        return {
            "score": 0,
            "feedback": "No answer was captured. Make sure your microphone is selected and working.",
            "improved_answer": "",
        }

    context = _history_block(history, limit=8)
    user_content = MOCK_EVALUATION_PROMPT.format(
        question=question.strip(), answer=answer,
    )
    if context:
        user_content = (
            "The exchange so far (use it to judge consistency and depth):\n"
            f"{context}\n\n{user_content}"
        )
    messages = [
        {"role": "system", "content": "You are an interview coach. Follow the output format exactly."},
        {"role": "user", "content": user_content},
    ]
    router = router or DEFAULT_ROUTER
    calls = {
        "groq": lambda: _chat(GROQ_TEXT_FALLBACK_MODEL, messages, max_tokens=1200, timeout=30),
        "gemini": lambda: _gemini_chat(messages, max_tokens=1200, timeout=30),
    }
    try:
        content, _served = router.run(calls)
    except Exception as exc:
        log.error("evaluate_mock_answer failed: %s", exc)
        return {"score": 0, "feedback": f"Evaluation failed: {exc}", "improved_answer": ""}

    score_match = re.search(r"SCORE:\s*(\d+)", content, re.IGNORECASE)
    feedback_match = re.search(r"FEEDBACK:\s*(.*?)(?=\nIMPROVED_ANSWER:|\Z)", content, re.IGNORECASE | re.DOTALL)
    improved_match = re.search(r"IMPROVED_ANSWER:\s*(.*)\Z", content, re.IGNORECASE | re.DOTALL)

    score = int(score_match.group(1)) if score_match else 0
    score = max(0, min(10, score))
    feedback = feedback_match.group(1).strip() if feedback_match else content.strip()
    improved = improved_match.group(1).strip() if improved_match else ""

    return {"score": score, "feedback": feedback, "improved_answer": improved}


# ---------------------------------------------------------------------------
# Screen + transcript fusion (coding interview copilot)
# ---------------------------------------------------------------------------

SOLVE_PROMPT_VARIANT = {
    "solve": (
        "Give the COMPLETE solution: full working code for the problem on screen."
    ),
    "next_step": (
        "Give only the IMMEDIATE next step: what to write or change right now to "
        "keep moving. code_solution must contain only that next fragment of code."
    ),
}

SOLVE_SYSTEM_PROMPT = """You are a senior engineer sitting next to a candidate during a live technical interview.

You receive screenshot(s) of what is on their screen (the problem statement, an editor with their half-finished solution, terminal/test output) plus the live transcript of what the interviewer just said. Understand BOTH: the problem on screen and what the interviewer is asking about.

{variant_instruction}

Respond with ONLY a JSON object, no markdown fences, exactly this shape:
{{"detected": true, "answer_text": "<how the candidate should respond OUT LOUD: spoken style, first person, <=80 words, no markdown>", "language": "<language of the code, or n/a>", "code_solution": "<complete working code, or empty string if no code applies>", "next_steps": ["<short step>", "<short step>"]}}

Rules:
- answer_text must read naturally if spoken aloud in an interview.
- code_solution must be complete and correct, ready to type; use the language already in the editor when visible.
- next_steps: 3-6 short bullets covering edge cases, complexity, or testing.
- If the screenshots show no coding problem, set detected to false and leave the other fields empty."""


def _parse_solve_response(text: str) -> dict:
    """Parse the fusion JSON (tolerates code fences and stray prose)."""
    result = {
        "detected": False, "answer_text": "", "language": "n/a",
        "code_solution": "", "next_steps": [],
    }
    if not text:
        return result
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        log.error("solve response is not JSON: %.200r", text)
        return result
    if not isinstance(data, dict):
        return result
    result["detected"] = bool(data.get("detected", True))
    result["answer_text"] = str(data.get("answer_text") or "").strip()
    result["language"] = str(data.get("language") or "n/a").strip()
    result["code_solution"] = str(data.get("code_solution") or "").strip()
    steps = data.get("next_steps")
    if isinstance(steps, list):
        result["next_steps"] = [str(s).strip() for s in steps if str(s).strip()]
    elif isinstance(steps, str) and steps.strip():
        result["next_steps"] = [line.strip("-* ").strip() for line in steps.splitlines() if line.strip()]
    if not result["answer_text"] and not result["code_solution"]:
        result["detected"] = False
    return result


def _solve_prompt(
    transcript: str,
    history: list[dict] | None,
    variant: str,
    problem_text: str | None = None,
    starter_code: str | None = None,
    has_images: bool = True,
) -> str:
    blocks = [SOLVE_SYSTEM_PROMPT.format(variant_instruction=SOLVE_PROMPT_VARIANT.get(variant, SOLVE_PROMPT_VARIANT["solve"]))]
    if not has_images:
        blocks.append(
            "No screenshot is attached this time. Rely entirely on the problem "
            "statement / starter code text below; when a problem statement is "
            "supplied it means a coding problem IS present (detected=true)."
        )
    if problem_text:
        blocks.append(
            "Problem statement (copied from the challenge panel, including every "
            "example and constraint):\n" + problem_text.strip()
        )
    if starter_code:
        blocks.append(
            "Starter code / editor content (copied from the challenge panel):\n"
            + starter_code.strip()
        )
    if history:
        recent = history[-4:]
        qa_lines = []
        for item in recent:
            qa_lines.append(f"Interviewer: {item.get('question', '')}")
            if item.get("answer"):
                qa_lines.append(f"Candidate: {item.get('answer', '')}")
        blocks.append("Recent exchange:\n" + "\n".join(qa_lines))
    blocks.append(f"Live transcript / latest interviewer statement:\n\"{(transcript or '').strip()}\"")
    return "\n\n".join(blocks)


def solve_with_screen(
    image_jpegs: list[bytes],
    transcript: str,
    history: list[dict] | None = None,
    variant: str = "solve",
    problem_text: str | None = None,
    starter_code: str | None = None,
    router: ProviderRouter | None = None,
) -> dict:
    """Fuse screen screenshot(s) + stashed problem/code text with the live
    transcript into a coding answer.

    Context (history, problem_text, starter_code, transcript) is rendered into
    one prompt BEFORE the provider is chosen; whichever stage wins — by
    preference or by failover — receives that identical payload.

    Stages run active-provider-first: vision (when images exist) for each
    provider, then text for each provider. The winning provider is reported as
    ``{groq|gemini}_{vision|text}`` and, when it differs from the active one,
    the router sticks to it (fallback) exactly like the single-call path.
    Raises RuntimeError if every stage fails.
    """
    router = router or DEFAULT_ROUTER
    images = [img for img in (image_jpegs or []) if img][:3]
    prompt = _solve_prompt(
        transcript, history, variant,
        problem_text=problem_text, starter_code=starter_code,
        has_images=bool(images),
    )
    order = router.stage_order()
    errors: list[str] = []

    def groq_vision() -> str:
        parts: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content, _msg = _groq_chat(
            GROQ_VISION_MODEL, [{"role": "user", "content": parts}],
            max_tokens=4000, timeout=60,
        )
        return content

    def gemini_vision() -> str:
        return _solve_with_gemini(images, prompt)

    def groq_text() -> str:
        return _chat(GROQ_CODE_MODEL, [{"role": "user", "content": prompt}], max_tokens=4000, timeout=60)

    def gemini_text() -> str:
        return _gemini_chat([{"role": "user", "content": prompt}], max_tokens=4000, timeout=60)

    vision_call = {"groq": groq_vision, "gemini": gemini_vision}
    text_call = {"groq": groq_text, "gemini": gemini_text}
    stages: list[tuple[str, str]] = []
    if images:
        stages.extend((provider, "vision") for provider in order)
    stages.extend((provider, "text") for provider in order)

    for provider, kind in stages:
        try:
            content = (vision_call if kind == "vision" else text_call)[provider]()
        except Exception as exc:
            errors.append(f"{provider} {kind} failed: {exc}")
            continue
        parsed = _parse_solve_response(content)
        if parsed["detected"]:
            parsed["provider"] = f"{provider}_{kind}"
            router.note_served(
                provider,
                failed=provider if provider != router.active else "",
                error=(errors[0] if errors else "")[:200],
            )
            return parsed
        errors.append(f"{provider} {kind} unusable output: {content[:120]!r}")

    raise RuntimeError("Screen solve failed: " + " | ".join(errors)[:400])


def _solve_with_gemini(image_jpegs: list[bytes], prompt: str) -> str:
    from google.genai import types as genai_types
    from gemini_client import _gemini_request_with_rotation, _get_client, _rate_limit

    _rate_limit()
    parts = [
        genai_types.Part.from_bytes(data=img, mime_type="image/jpeg")
        for img in image_jpegs
    ]
    parts.append(prompt)
    response = _gemini_request_with_rotation(
        lambda: _get_client().models.generate_content(
            model=get_settings()["MODEL_NAME"], contents=parts,
        )
    )
    return response.text or ""

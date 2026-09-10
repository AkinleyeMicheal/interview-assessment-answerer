# Quizbox AI

A phone-based real-time quiz answerer that captures questions via camera and answers them using multiple AI backends (OpenRouter, Google Antigravity, Gemini, Ollama). Built with FastAPI + vanilla JS, designed for mobile-first use over HTTPS.

## Features

- **Phone camera capture** — uses `getUserMedia()` with rear-facing camera, no app install needed
- **Multiple AI providers** — OpenRouter, Google Antigravity (managed agent), Gemini Vision, Ollama (local)
- **Auto-fallback chain** — automatically rotates through providers on failure or rate limit
- **OCR + LLM mode** — Tesseract extracts text, LLM answers (cheaper, more reliable for clear text)
- **Gemini API key rotation** — supports 3 Gemini keys with auto-rotate on rate limit + manual rotate
- **Capture cooldown timer** — configurable countdown between captures to prevent rate limiting
- **Self-signed HTTPS** — required for camera access on phones
- **QR code** — scan to open on any device on the same WiFi
- **MCQ + Essay** — handles both multiple-choice and open-ended questions
- **Settings modal** — configure all API keys, models, and camera URL from the phone UI

## Requirements

- Python 3.10+
- Tesseract OCR installed ([Windows](https://github.com/UB-Mannheim/tesseract/wiki), `apt install tesseract-ocr` on Linux)
- Ollama (optional, for local models)
- At least one API key: OpenRouter (free tier available) or Gemini

## Installation

```bash
git clone https://github.com/psycho237-prog/Quizbox-AI-
cd Quizbox-AI-
pip install -r requirements.txt
```

### Install Tesseract (Windows)

```bash
winget install UB-Mannheim.TesseractOCR
```

The path `C:\Program Files\Tesseract-OCR\tesseract.exe` is used by default.

## Setup

1. Copy `.env.example` to `.env` and fill in your API keys:

```env
GEMINI_API_KEY=your_gemini_key
GEMINI_API_KEY_2=          # optional, for rotation
GEMINI_API_KEY_3=          # optional, for rotation
OPENROUTER_API_KEY=your_openrouter_key
OCR_MODE=openrouter        # default mode
MODEL_NAME=gemini-2.5-flash
OPENROUTER_MODEL_1=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
OPENROUTER_MODEL_2=openrouter/free
OLLAMA_MODEL=granite3.2-vision:2b
OLLAMA_URL=http://localhost:11434
CAMERA_URL=http://your-ip-cam:8080/video
```

2. Generate SSL certs (auto-generated on first run, or manually):

```bash
python gen_cert.py
```

3. Run the server:

```bash
python server.py
```

4. Open the URL printed in the terminal on your phone (e.g., `https://192.168.1.100:8000`), or scan the QR code from the app header.

## Usage

1. Open the app on your phone via the URL or QR code
2. Grant camera permission when prompted
3. Select a mode from the mode bar:
   - **OpenRouter** — vision model via OpenRouter (fastest, free tier)
   - **OCR + OpenRouter** — Tesseract extracts text, OpenRouter answers (cheapest)
   - **Antigravity** — Google's managed AI agent (most capable)
   - **Auto Fallback** — tries all providers in order, skips rate-limited ones
   - **Gemini** — direct Gemini Vision API
   - **Ollama** — local model (OCR mode, requires Ollama running)
4. Point your phone camera at a quiz question
5. Tap **Capture** — the answer appears in the panel below
6. The capture button shows a cooldown timer (default 20s) — tap the clock icon to adjust
7. After cooldown, tap Capture again for the next question

## Modes

| Mode | How it works | Requires |
|------|-------------|----------|
| `openrouter` | Sends image directly to OpenRouter vision model | OpenRouter key |
| `tesseract_openrouter` | OCR extracts text, sends to OpenRouter text model | OpenRouter key |
| `antigravity` | Google's managed AI agent with reasoning | Gemini key |
| `auto` | Tries all providers in order, skipping rate-limited ones | Any key |
| `gemini_vision` | Sends image directly to Gemini Vision API | Gemini key |
| `ollama_text` | OCR + local Ollama model | Ollama running |

## Auto-Fallback Chain

When `auto` mode is active, providers are tried in this order:

1. OCR + OpenRouter (primary)
2. OpenRouter vision (direct)
3. OCR + OpenRouter (backup model)
4. OCR + Gemini
5. Gemini vision
6. OCR + Ollama (local)
7. Ollama vision (local)

If a provider is rate-limited, it's skipped for 60 seconds. Gemini keys auto-rotate on rate limit.

## Gemini Key Rotation

The app supports up to 3 Gemini API keys for rotation:

- **Auto-rotate**: When a key hits rate limit, the next key is used automatically
- **Manual rotate**: Tap the key icon in the header to switch keys
- **Toast notifications**: "Key 1 rate limited, switching to key 2..." on auto-rotate

Configure keys in `.env` or via the Settings modal on your phone.

## Capture Cooldown

After answering a question, the capture button shows a countdown (e.g., `Wait 20s`) and is disabled until the timer expires. This prevents hitting rate limits.

- Default: 20 seconds
- Adjust via the clock icon next to the capture button
- Persisted in `localStorage` across sessions

## Architecture

```
quizbox/
├── server.py              # FastAPI backend (HTTPS, REST endpoints)
├── gemini_client.py       # All AI providers (OpenRouter, Antigravity, Gemini, Ollama, Tesseract)
├── config.py              # Settings management + Gemini key rotation
├── gen_cert.py            # Self-signed SSL certificate generator
├── static/
│   ├── index.html         # Phone-first UI (capture, answer panel, settings, QR)
│   └── qr.js              # QR code generator
├── .env                   # API keys and configuration
├── cert.pem / key.pem     # SSL certificates (auto-generated)
└── requirements.txt       # Python dependencies
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the phone UI |
| `POST` | `/api/accept` | Accept a base64 image, return answer |
| `GET` | `/api/settings` | Get current settings |
| `POST` | `/api/settings` | Update settings |
| `GET` | `/api/providers` | Get provider status and cooldowns |
| `GET` | `/api/gemini-key/status` | Get Gemini key rotation status |
| `POST` | `/api/gemini-key/rotate` | Manually rotate to next Gemini key |
| `GET` | `/api/camera-status` | Check if IP camera is connected |
| `GET` | `/api/ollama-status` | Check if Ollama is running |

## Configuration

All settings are in `.env` and can be updated from the phone UI via the Settings modal.

| Key | Default | Description |
|-----|---------|-------------|
| `GEMINI_API_KEY` | — | Primary Gemini API key |
| `GEMINI_API_KEY_2` | — | Secondary key for rotation |
| `GEMINI_API_KEY_3` | — | Tertiary key for rotation |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (free tier available) |
| `OCR_MODE` | `openrouter` | Default analysis mode |
| `MODEL_NAME` | `gemini-2.5-flash` | Gemini model name |
| `OPENROUTER_MODEL_1` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | Primary OpenRouter model |
| `OPENROUTER_MODEL_2` | `openrouter/free` | Backup OpenRouter model |
| `OLLAMA_MODEL` | `granite3.2-vision:2b` | Local Ollama model |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `CAMERA_URL` | — | IP Webcam stream URL (optional, phone camera used by default) |

## License

MIT License. Feel free to use, modify, and share.

## Author

Micheal Akinleye

# Quizbox AI

Quizbox sends a selected desktop region to a phone over the local network and answers captured questions using one of two explicit modes.

## Modes

| Phone tab | Pipeline |
| --- | --- |
| **Groq Vision** | Qwen 3.6 Vision on Groq reads the screenshot and preserves the question/options. Groq Compound Mini then uses Groq's built-in web search to verify and return the answer. No local search service is used. |
| **Gemini Vision** | Gemini receives the screenshot directly and returns the extracted question, options, answer, and explanation. |

The selected tab is included with every capture request, so it cannot be routed to the other provider by a delayed settings update.

## Setup

1. Copy `.env.example` to `.env`.
2. Set `GROQ_API_KEY` for Groq Vision and/or `GEMINI_API_KEY` for Gemini Vision.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the HTTPS server:

   ```bash
   python server.py
   ```

5. Open the printed URL on your phone, select a desktop region, choose a tab, and tap **Capture**.

## Notes

- Groq Vision uses Groq Compound's server-side web-search tool; SearXNG is not required or queried.
- Gemini supports up to three keys. The app rotates to the next configured key if it receives a rate-limit error.
- The phone pulls only the latest desktop JPEG, preventing stale live-feed buffering on slow Wi-Fi.

## Tests

Run the local end-to-end stream checks with:

```bash
python -m unittest -v tests.test_live_stream
```

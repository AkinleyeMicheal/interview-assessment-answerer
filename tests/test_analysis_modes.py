"""Provider-routing and payload tests; all external API calls are mocked."""

import asyncio
import base64
import unittest
from unittest.mock import patch

import gemini_client
import server


class AnalysisModeTests(unittest.TestCase):
    def test_parser_preserves_multiline_options(self):
        parsed = gemini_client._parse_response(
            "QUESTION: Which value is prime?\n"
            "TYPE: MCQ\n"
            "OPTIONS: A. 21\nB. 29\nC. 35\n"
            "ANSWER: B. 29\n"
            "EXPLANATION: 29 has no positive divisors except 1 and itself."
        )
        self.assertTrue(parsed["detected"])
        self.assertEqual(parsed["type"], "MCQ")
        self.assertIn("B. 29", parsed["options"])
        self.assertEqual(parsed["answer"], "B. 29")

    @patch("gemini_client._call_groq_vision")
    @patch("gemini_client._groq_chat")
    def test_groq_mode_extracts_then_uses_groq_compound_search(self, groq_chat, call_vision):
        call_vision.return_value = (
            "QUESTION: What is the capital of France?\n"
            "TYPE: MCQ\n"
            "OPTIONS: A. Rome\nB. Paris\nC. Madrid"
        )
        groq_chat.return_value = (
            "QUESTION: What is the capital of France?\n"
            "TYPE: MCQ\n"
            "OPTIONS: A. Rome\nB. Paris\nC. Madrid\n"
            "ANSWER: B. Paris\n"
            "EXPLANATION: Paris is France's capital.",
            {"executed_tools": [{"type": "search", "output": "Source https://example.org/france"}]},
        )

        result = gemini_client.analyze_image(b"not-used", "groq_vision")

        self.assertEqual(result["provider"], "groq_vision")
        self.assertEqual(result["answer"], "B. Paris")
        self.assertTrue(result["search_used"])
        self.assertEqual(result["sources"][0]["uri"], "https://example.org/france")
        args, kwargs = groq_chat.call_args
        self.assertEqual(args[0], gemini_client.GROQ_SEARCH_MODEL)
        self.assertTrue(kwargs["compound"])
        self.assertIn("B. Paris", args[1][0]["content"])
        self.assertIn("web-search tool", args[1][0]["content"])

    @patch("gemini_client._call_groq_vision")
    @patch("gemini_client._groq_chat")
    def test_groq_mode_returns_an_answer_when_compound_is_temporarily_unavailable(self, groq_chat, call_vision):
        call_vision.return_value = "QUESTION: 2 + 2?\nTYPE: MCQ\nOPTIONS: A. 3\nB. 4"
        groq_chat.side_effect = [
            RuntimeError("Groq request failed (HTTP 413)"),
            ("QUESTION: 2 + 2?\nTYPE: MCQ\nOPTIONS: A. 3\nB. 4\nANSWER: B. 4\nEXPLANATION: Arithmetic.", {}),
        ]

        result = gemini_client.analyze_image(b"not-used", "groq_vision")

        self.assertEqual(result["answer"], "B. 4")
        self.assertFalse(result["search_used"])
        self.assertIn("HTTP 413", result["search_error"])
        self.assertEqual(groq_chat.call_args.args[0], gemini_client.GROQ_TEXT_FALLBACK_MODEL)

    @patch("gemini_client._rate_limit")
    @patch("gemini_client._get_client")
    @patch("gemini_client.get_settings")
    def test_gemini_mode_sends_the_image_only_to_gemini(self, get_settings, get_client, _rate_limit):
        get_settings.return_value = {"MODEL_NAME": "gemini-test-model"}

        class FakeModels:
            def generate_content(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {
                    "text": "QUESTION: 2 + 2?\nTYPE: MCQ\nOPTIONS: A. 3\nB. 4\nANSWER: B. 4\nEXPLANATION: Arithmetic."
                })()

        fake_client = type("Client", (), {"models": FakeModels()})()
        get_client.return_value = fake_client

        result = gemini_client.analyze_image(b"jpeg", "gemini_vision")

        self.assertEqual(result["provider"], "gemini_vision")
        self.assertEqual(result["answer"], "B. 4")
        self.assertEqual(fake_client.models.kwargs["model"], "gemini-test-model")

    def test_capture_request_mode_overrides_persisted_default(self):
        image = base64.b64encode(b"jpeg").decode("ascii")
        with patch("server.analyze_image", return_value={"detected": False, "provider": "gemini_vision"}) as analyze:
            response = asyncio.run(server.accept_image(server.CaptureRequest(image=image, mode="gemini_vision")))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(analyze.call_args.args[1], "gemini_vision")

    def test_provider_status_uses_the_actual_groq_provider_key(self):
        providers = {
            "groq_vision": {"name": "Groq", "configured": True},
            "gemini_vision": {"name": "Gemini", "configured": True},
        }
        with patch("server.get_provider_status", return_value=providers), patch(
            "server.check_groq_status", return_value={"valid": True}
        ):
            result = asyncio.run(server.providers_status())

        self.assertTrue(result["groq_vision"]["groq_status"]["valid"])


if __name__ == "__main__":
    unittest.main()

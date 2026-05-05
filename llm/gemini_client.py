import os, json, traceback
from typing import Dict, Any, List, Optional
import backoff
import httpx


GEMINI_API_KEY_ENV = "GOOGLE_API_KEY"
GEMINI_MODEL = "models/gemini-2.5-pro"
GEMINI_ENDPOINT_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL):
        self.api_key = api_key or os.environ.get(GEMINI_API_KEY_ENV, "")
        self.model = model
        if not self.api_key:
            raise RuntimeError(f"Missing {GEMINI_API_KEY_ENV} for Gemini API")

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        params = {"key": self.api_key}
        try:
            with httpx.Client(timeout=60) as client:
                url = f"{GEMINI_ENDPOINT_BASE}/{self.model}:generateContent"
                r = client.post(url, params=params, json=payload)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            print(f"Gemini API error: {e}")
            traceback.print_exc()
            raise

    def complete_json(self, system_prompt: str, user_prompt: str, schema_hint: Optional[str] = None, temperature: float = 0.2) -> Dict[str, Any]:
        """
        Ask Gemini to produce ONLY valid JSON. We instruct it strictly and parse.
        """
        instructions = (
            f"You are a precise information extraction system. {system_prompt}\n"
            "Return ONLY valid minified JSON. No prose, no markdown, no code fences."
        )
        if schema_hint:
            instructions += f"\nJSON schema hint: {schema_hint}"
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": instructions + "\n\n" + user_prompt}]}
            ],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096}
        }
        resp = self._post(payload)
        try:
            text = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
            # Unwrap code fences if present
            if text.startswith("```"):
                # Remove leading ``` or ```json
                first_newline = text.find("\n")
                if first_newline != -1:
                    fence_lang = text[3:first_newline].strip()
                    text = text[first_newline+1:]
                text = text.rstrip("`\n ")
            return json.loads(text)
        except Exception as e:
            print(f"Failed to parse JSON from Gemini: {e}")
            traceback.print_exc()
            raise

    def complete_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]}
            ],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max(4096, max_tokens)}
        }
        resp = self._post(payload)
        try:
            return resp["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"Failed to decode Gemini text: {e}")
            traceback.print_exc()
            raise

    def healthcheck(self) -> bool:
        try:
            txt = self.complete_text(
                "You are a healthcheck utility.",
                "Reply with the single word: OK",
                temperature=0.0,
                max_tokens=64,
            )
            return "OK" in txt.upper()
        except Exception:
            return False
import os
import traceback
from typing import Dict, Any, Optional

import backoff
import httpx


OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-5"):
        self.api_key = api_key or os.environ.get(OPENROUTER_API_KEY_ENV, "")
        self.model = model
        if not self.api_key:
            raise RuntimeError(f"Missing {OPENROUTER_API_KEY_ENV} for OpenRouter API")

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter recommends one of these for attribution
            "X-Title": "resume_validity_builder",
        }
        try:
            with httpx.Client(timeout=120) as client:
                r = client.post(OPENROUTER_ENDPOINT, headers=headers, json=payload)
                r.raise_for_status()
                try:
                    return r.json()
                except Exception:
                    # Log non-JSON body for debugging
                    try:
                        print("[ERROR] OpenRouter non-JSON response body (truncated to 2KB):")
                        txt = r.text
                        print(txt[:2048])
                    except Exception:
                        pass
                    raise
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            print(traceback.format_exc())
            raise

    def complete_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = self._post(payload)
        try:
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Failed to decode OpenRouter text: {e}")
            print(traceback.format_exc())
            raise



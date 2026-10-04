"""
LLM Client for OpenRouter
Provides access to OpenRouter's free and commercial models using standard REST API.
"""
import os
import requests
from typing import Dict, Any, Optional

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

POPULAR_FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free"
]


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip() and self.api_key.strip() != "your_openrouter_api_key_here")

    def call_model(
        self,
        prompt: str,
        model: str = "meta-llama/llama-3.3-70b-instruct:free",
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> Dict[str, Any]:
        """
        Sends a completion request to OpenRouter.
        """
        if not self.is_configured():
            return {
                "success": False,
                "error": "OpenRouter API Key is not set. Please provide it in .env or via the sidebar.",
                "output": ""
            }

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Prompt_Tune",
            "Content-Type": "application/json"
        }

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                return {
                    "success": True,
                    "output": content,
                    "usage": data.get("usage", {}),
                    "model_used": data.get("model", model)
                }
            else:
                try:
                    err_json = response.json()
                    err_msg = err_json.get("error", {}).get("message", response.text)
                except Exception:
                    err_msg = response.text
                return {
                    "success": False,
                    "error": f"API Error (HTTP {response.status_code}): {err_msg}",
                    "output": ""
                }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "The OpenRouter request timed out after 60 seconds.",
                "output": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Network / Client Exception: {str(e)}",
                "output": ""
            }

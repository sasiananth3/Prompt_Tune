"""
LLM Client for OpenRouter
Provides access to OpenRouter's free models using standard REST API,
with dynamic discovery of currently available free models.
"""
import os
import requests
from typing import Dict, Any, List, Optional

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

DEFAULT_FALLBACK_FREE_MODELS = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "cohere/north-mini-code:free"
]


def fetch_live_free_models() -> List[str]:
    """
    Dynamically queries OpenRouter API to fetch all currently active free models.
    """
    try:
        res = requests.get(OPENROUTER_MODELS_URL, timeout=8)
        if res.status_code == 200:
            data = res.json().get("data", [])
            free_models = []
            # Always put openrouter/free first if present
            for m in data:
                m_id = m.get("id", "")
                pricing = m.get("pricing", {})
                is_free = ":free" in m_id or (pricing.get("prompt") == "0" and pricing.get("completion") == "0")
                if m_id == "openrouter/free":
                    free_models.insert(0, m_id)
                elif is_free:
                    free_models.append(m_id)
            if "openrouter/free" not in free_models:
                free_models.insert(0, "openrouter/free")
            if free_models:
                return free_models
    except Exception:
        pass
    return DEFAULT_FALLBACK_FREE_MODELS


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip() and self.api_key.strip() != "your_openrouter_api_key_here")

    def call_model(
        self,
        prompt: str,
        model: str = "openrouter/free",
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
                choices = data.get("choices", [])
                if not choices:
                    return {
                        "success": False,
                        "error": "The model returned an empty response. This happens when the free endpoint is temporarily overloaded. Please retry or pick a different model from the sidebar.",
                        "output": ""
                    }
                choice = choices[0]
                message = choice.get("message", {})
                content = message.get("content") or ""
                reasoning = message.get("reasoning") or ""
                finish_reason = choice.get("finish_reason")

                # If content is empty/None but reasoning exists (e.g. reasoning models like DeepSeek-R1)
                if not content.strip() and reasoning.strip():
                    content = reasoning.strip()

                if not content.strip():
                    return {
                        "success": False,
                        "error": "Model finished without generating readable text (upstream endpoint returned empty output). Try increasing Max Tokens or selecting a different free model from the sidebar.",
                        "output": ""
                    }

                # Warn user if output was cut off prematurely
                if finish_reason == "length":
                    content += "\n\n---\n> ⚠️ **Output Truncated:** The model reached the `Max Tokens` limit before finishing. Expand **⚙️ Generation Parameters** in the sidebar and increase **Max Tokens** (e.g. to 3000+) to get the complete output."

                return {
                    "success": True,
                    "output": content,
                    "usage": data.get("usage", {}),
                    "model_used": data.get("model", model),
                    "finish_reason": finish_reason
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

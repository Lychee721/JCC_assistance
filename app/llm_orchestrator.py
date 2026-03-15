from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"


class LLMOrchestrator:
    """Gemini-only orchestrator using the OpenAI-compatible Gemini endpoint."""

    def __init__(
        self,
        system_prompt_path: str | Path = "prompts/item_recommendation_system.txt",
        user_template_path: str | Path = "prompts/item_recommendation_user_template.txt",
        model: str | None = None,
        temperature: float = 0.45,
        max_tokens: int = 700,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_history_messages: int = 12,
    ) -> None:
        self.system_prompt_path = Path(system_prompt_path)
        self.user_template_path = Path(user_template_path)
        self.provider = "gemini"
        self.model = model or self._env_first("GEMINI_MODEL", default=DEFAULT_GEMINI_MODEL)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = (api_base or self._env_first("GEMINI_API_BASE", default=DEFAULT_GEMINI_API_BASE)).rstrip("/")
        self.api_key = api_key or self._env_first("GEMINI_API_KEY", "GOOGLE_API_KEY", default="")
        self.timeout_seconds = timeout_seconds
        self.max_history_messages = max_history_messages

    def _env_first(self, *keys: str, default: str = "") -> str:
        for key in keys:
            value = os.getenv(key)
            if value and value.strip():
                return value.strip()
        return default

    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def build_messages(self, request: dict[str, Any], recommendation_payload: dict[str, Any]) -> list[dict[str, str]]:
        system_prompt = self._read(self.system_prompt_path)
        template = self._read(self.user_template_path)
        user_prompt = (
            template.replace("{{ user_question }}", request.get("user_question") or "")
            .replace("{{ intent }}", request.get("intent", "balanced"))
            .replace("{{ target_champion }}", request.get("target_champion") or "")
            .replace("{{ stage }}", request.get("stage") or "")
            .replace("{{ components_json }}", json.dumps(request.get("components", {}), ensure_ascii=False))
            .replace(
                "{{ recommendation_payload_json }}",
                json.dumps(recommendation_payload, ensure_ascii=False, indent=2),
            )
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def _trim_history(self, chat_history: list[dict[str, str]] | None) -> list[dict[str, str]]:
        if not chat_history:
            return []
        normalized: list[dict[str, str]] = []
        for item in chat_history:
            role = item.get("role")
            content = item.get("content", "")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                normalized.append({"role": role, "content": content.strip()})
        return normalized[-self.max_history_messages :]

    def _build_request_payload(self, conversation: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "temperature": self.temperature,
        }
        if os.getenv("GEMINI_SEND_MAX_TOKENS", "0").strip().lower() in {"1", "true"} and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        return payload

    def generate_response(
        self,
        request: dict[str, Any],
        recommendation_payload: dict[str, Any],
        user_question: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, str]:
        messages = self.build_messages(request, recommendation_payload)
        conversation: list[dict[str, str]] = [messages[0], messages[1]]
        conversation.extend(self._trim_history(chat_history))
        if user_question:
            conversation.append({"role": "user", "content": user_question})

        fallback = recommendation_payload.get("answer_text", "")
        if not self.is_enabled():
            return {
                "text": fallback,
                "source": "fallback",
                "error": "missing_api_key(provider=gemini)",
            }

        try:
            payload = self._build_request_payload(conversation)
            with httpx.Client(timeout=self.timeout_seconds) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                try:
                    response = client.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    if exc.response is not None and exc.response.status_code in {401, 403}:
                        alt_headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
                        response = client.post(f"{self.api_base}/chat/completions", headers=alt_headers, json=payload)
                        response.raise_for_status()
                    else:
                        raise

            response_json = response.json()
            content = self._extract_content(response_json)
            if content:
                return {"text": content, "source": "llm", "error": ""}
            return {"text": fallback, "source": "fallback", "error": "empty_llm_content"}
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = ""
            try:
                body = (exc.response.text or "").strip()
            except Exception:
                body = ""
            return {
                "text": fallback,
                "source": "fallback",
                "error": f"HTTPStatusError(provider=gemini,status={status},body={body[:700]})",
            }
        except Exception as exc:
            return {
                "text": fallback,
                "source": "fallback",
                "error": f"{type(exc).__name__}(provider=gemini): {exc}",
            }

    def generate_text(
        self,
        request: dict[str, Any],
        recommendation_payload: dict[str, Any],
        user_question: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        result = self.generate_response(
            request,
            recommendation_payload,
            user_question=user_question,
            chat_history=chat_history,
        )
        return result["text"]

    def _extract_content(self, response_json: dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks).strip()
        return ""

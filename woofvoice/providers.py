"""Small dependency-free streaming clients for common chat APIs."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from typing import Protocol


Message = dict[str, str]


class ChatBackend(Protocol):
    """Interface implemented by a provider that streams text tokens."""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        ...


def _post(url: str, payload: dict[str, object], headers: dict[str, str]) -> urllib.request.Request:
    request_headers = {"Content-Type": "application/json", **headers}
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=request_headers,
        method="POST",
    )


def _url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


class OpenAIChatBackend:
    """Stream OpenAI Chat Completions and compatible APIs."""

    def __init__(self, base_url: str, api_key: str | None, model: str) -> None:
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url = _url(self.url, "chat/completions")
        self.api_key = api_key
        self.model = model

    def stream(self, messages: list[Message]) -> Iterator[str]:
        headers = {"Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = _post(
            self.url,
            {"model": self.model, "messages": messages, "stream": True},
            headers,
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                event = _json_event(data, "OpenAI")
                choices = event.get("choices")
                if choices and isinstance(choices[0], dict):
                    delta = choices[0].get("delta", {})
                    if isinstance(delta, dict) and delta.get("content"):
                        yield str(delta["content"])


class OllamaChatBackend:
    """Stream Ollama's native newline-delimited `/api/chat` response."""

    def __init__(self, base_url: str, model: str) -> None:
        self.url = _url(base_url, "api/chat")
        self.model = model

    def stream(self, messages: list[Message]) -> Iterator[str]:
        request = _post(self.url, {"model": self.model, "messages": messages, "stream": True}, {})
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                event = _json_event(line, "Ollama")
                message = event.get("message", {})
                if isinstance(message, dict) and message.get("content"):
                    yield str(message["content"])
                if event.get("done"):
                    return


class AnthropicChatBackend:
    """Stream Anthropic Messages API text deltas."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 1024,
    ) -> None:
        self.url = _url(base_url, "v1/messages")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    def stream(self, messages: list[Message]) -> Iterator[str]:
        request = _post(
            self.url,
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages,
                "stream": True,
            },
            {
                "Accept": "text/event-stream",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                event = _json_event(line[5:].strip(), "Anthropic")
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if isinstance(delta, dict) and delta.get("text"):
                        yield str(delta["text"])
                if event.get("type") == "message_stop":
                    return


def _json_event(data: str, provider: str) -> dict[str, object]:
    try:
        event = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid {provider} streaming event: {data!r}") from exc
    if not isinstance(event, dict):
        raise RuntimeError(f"Invalid {provider} streaming event: expected an object")
    return event

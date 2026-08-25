"""Model backends — ingredient #1: a swappable model the engine doesn't care about.

A Backend turns a list of chat messages into a string plus token usage. The
engine only ever touches this interface, so swapping OpenRouter for a local model
or a deterministic mock is a one-line change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Usage:
    """Token/-call accounting returned by every backend call."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Backend:
    """Interface: messages -> (text, Usage)."""

    def complete(self, messages: list[dict], *, model: str | None = None) -> tuple[str, Usage]:
        raise NotImplementedError


class OpenRouterBackend(Backend):
    """The default backend. Speaks the OpenAI-compatible chat-completions API, so
    it works with OpenRouter (default), or any compatible endpoint via RLM_BASE_URL.
    """

    def __init__(
        self,
        model: str = "openai/gpt-5-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        timeout: int = 120,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = (base_url or os.environ.get("RLM_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "No API key. Set OPENROUTER_API_KEY in the environment, or pass api_key=."
            )

    def complete(self, messages, *, model=None):
        import requests  # imported lazily so the mock path needs no network deps

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        text = body["choices"][0]["message"].get("content") or ""
        u = body.get("usage") or {}
        return text, Usage(
            calls=1,
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
        )


class MockBackend(Backend):
    """A deterministic backend for testing the engine loop with no network.

    - `root_script`: the assistant turns the *root* agent plays, in order.
    - any request whose latest user message contains "CHUNK" is treated as a
      sub-agent and answered with `sub_answer` (it should FINAL immediately).
    - `tool_answers`: substring -> canned reply, used for plain llm() calls.
    """

    def __init__(
        self,
        root_script: list[str],
        sub_answer: str = "```python\nFINAL('3')\n```",
        tool_answers: dict[str, str] | None = None,
    ):
        self.root_script = root_script
        self.sub_answer = sub_answer
        self.tool_answers = tool_answers or {}
        self._i = 0

    def complete(self, messages, *, model=None):
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        u = Usage(calls=1, prompt_tokens=5, completion_tokens=5)

        if "CHUNK" in last_user:
            return self.sub_answer, u
        for needle, answer in self.tool_answers.items():
            if needle in last_user:
                return answer, u

        text = self.root_script[self._i] if self._i < len(self.root_script) else "```python\nFINAL(None)\n```"
        self._i += 1
        return text, u

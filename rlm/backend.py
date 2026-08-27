"""Model backends — ingredient #1: a swappable model the engine doesn't care about.

A Backend turns a list of chat messages into a string plus token usage. The
engine only ever touches this interface, so swapping DeepSeek for OpenRouter, a
local server, or a deterministic mock is a one-line change.

DeepSeek, OpenRouter, OpenAI, and most local servers all speak the same
OpenAI-compatible `/chat/completions` API, so they share one implementation
(`OpenAICompatBackend`) and differ only in base URL, default model, and which env
var holds the key. Presets are provided for the common ones.
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


class OpenAICompatBackend(Backend):
    """Any provider that speaks the OpenAI-compatible chat-completions API.

    Configure via subclass presets (below) or directly. Key resolution order:
    explicit `api_key=` -> the provider's env var -> the generic `RLM_API_KEY`.
    Base URL can always be overridden with the `RLM_BASE_URL` env var.
    """

    provider: str = "openai-compat"
    default_base_url: str | None = None
    default_model: str | None = None
    env_key: str | None = None  # provider-specific env var holding the API key

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        timeout: int = 120,
        env_key: str | None = None,
    ):
        env_key = env_key or self.env_key
        self.model = model or self.default_model
        self.base_url = (
            base_url or os.environ.get("RLM_BASE_URL") or self.default_base_url or ""
        ).rstrip("/")
        self.api_key = (
            api_key
            or (os.environ.get(env_key) if env_key else None)
            or os.environ.get("RLM_API_KEY")
        )
        self.temperature = temperature
        self.timeout = timeout

        if not self.base_url:
            raise RuntimeError("No base_url set for the backend.")
        if not self.model:
            raise RuntimeError("No model set. Pass model= or set a provider default.")
        if not self.api_key:
            want = env_key or "RLM_API_KEY"
            raise RuntimeError(f"No API key. Set {want} in the environment, or pass api_key=.")

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


class DeepSeekBackend(OpenAICompatBackend):
    """DeepSeek's native API. Models: `deepseek-chat` (V3), `deepseek-reasoner` (R1)."""

    provider = "deepseek"
    default_base_url = "https://api.deepseek.com"
    default_model = "deepseek-chat"
    env_key = "DEEPSEEK_API_KEY"


class OpenRouterBackend(OpenAICompatBackend):
    """OpenRouter — one key, many models (e.g. `openai/gpt-5-mini`)."""

    provider = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    default_model = "openai/gpt-5-mini"
    env_key = "OPENROUTER_API_KEY"


# Registry so the CLI (and later the harness) can pick a provider by name.
BACKENDS: dict[str, type[OpenAICompatBackend]] = {
    "deepseek": DeepSeekBackend,
    "openrouter": OpenRouterBackend,
    "openai-compat": OpenAICompatBackend,
}

DEFAULT_PROVIDER = os.environ.get("RLM_PROVIDER", "deepseek")


def make_backend(provider: str | None = None, model: str | None = None, **kwargs) -> Backend:
    """Build a backend by provider name (defaults to DeepSeek / $RLM_PROVIDER)."""
    name = (provider or DEFAULT_PROVIDER).lower()
    try:
        cls = BACKENDS[name]
    except KeyError:
        raise RuntimeError(
            f"Unknown provider {name!r}. Choose from: {', '.join(BACKENDS)}."
        ) from None
    return cls(model=model, **kwargs)


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

"""Model backends behind one method: `complete(messages) -> str`."""
from __future__ import annotations

import os
from typing import Callable, Protocol

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class LLM(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


class ScriptedLLM:
    """Replays a fixed list of replies. Makes agent logic fully testable without a model."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            raise RuntimeError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)


class FunctionLLM:
    """Wrap any callable, e.g. a local model server client."""

    def __init__(self, fn: Callable[[list[Message]], str]):
        self.fn = fn

    def complete(self, messages: list[Message]) -> str:
        return self.fn(messages)


class AnthropicLLM:  # pragma: no cover - network
    """Claude backend (`pip install anthropic`, ANTHROPIC_API_KEY)."""

    def __init__(self, model: str, max_tokens: int = 1024):
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model, self.max_tokens = model, max_tokens

    def complete(self, messages: list[Message]) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        out = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system, messages=convo)
        return "".join(b.text for b in out.content if getattr(b, "type", "") == "text")

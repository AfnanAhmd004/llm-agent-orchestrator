"""Typed tools with argument validation."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

_TYPES = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    description: str
    params: dict[str, type]

    def spec(self) -> str:
        args = ", ".join(f"{k}: {_TYPES.get(t, 'any')}" for k, t in self.params.items())
        return f"- {self.name}({args}): {self.description}"

    def __call__(self, args: dict) -> Any:
        missing = set(self.params) - set(args)
        extra = set(args) - set(self.params)
        if missing or extra:
            raise ValueError(f"{self.name}: missing {sorted(missing)} unexpected {sorted(extra)}")
        clean = {}
        for k, t in self.params.items():
            v = args[k]
            if t is float and isinstance(v, int) and not isinstance(v, bool):
                v = float(v)
            if not isinstance(v, t):
                raise TypeError(f"{self.name}.{k} expects {t.__name__}, got {type(v).__name__}")
            clean[k] = v
        return self.fn(**clean)


def tool(description: str):
    """Decorator: build a Tool from a type-annotated function."""

    def wrap(fn: Callable) -> Tool:
        sig = inspect.signature(fn)
        params = {n: p.annotation for n, p in sig.parameters.items()}
        return Tool(fn.__name__, fn, description, params)

    return wrap
